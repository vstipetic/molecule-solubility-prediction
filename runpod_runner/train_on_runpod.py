"""Orchestrator: train any of our models on a transient RunPod GPU pod.

Spins up one on-demand GPU pod, uploads data (+ optional extra artifacts such
as a pretrained checkpoint), runs the selected training module over SSH,
streams the training log to the console, then SCPs the checkpoint and log back
to your machine and terminates the pod. Teardown happens in a ``try/finally``
so the pod is always released even if training fails.

Random Forest is intentionally not supported here — per the project plan it
runs locally on CPU.

Required environment:
    RUNPOD_API_KEY    RunPod API key (RunPod -> Settings -> API Keys).
    WANDB_API_KEY     Weights & Biases API key, injected into the pod so the
                      run logs remotely. If unset, the pod falls back to wandb
                      offline mode (and you still get the SCP'd checkpoint/log).

Required SSH setup (one-time):
    Add your SSH **public** key under RunPod -> Settings -> SSH Keys. RunPod
    injects it into the pod automatically. Point the orchestrator at the
    matching private key with --ssh-private-key (default ~/.ssh/id_ed25519).

Example:
    # Train ChemBERTa (uploads Storage/Datasets/splits/ by default)
    uv run --extra runpod python -m runpod_runner.train_on_runpod \\
        --model chemberta \\
        --training-args "--freeze-encoder --n-epochs 50"

    # Fine-tune the scratch transformer (upload its pretrained checkpoint too)
    uv run --extra runpod python -m runpod_runner.train_on_runpod \\
        --model transformer \\
        --upload checkpoints/pretrained_model.pt:/workspace/uploads/pretrained.pt \\
        --training-args "--pretrained-path /workspace/uploads/pretrained.pt"
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List, Optional, Tuple

from runpod_runner.pod_manager import DEFAULT_GPU_TYPE, DEFAULT_IMAGE, PodManager
from Train.wandb_utils import load_dotenv

try:
    import paramiko
except ImportError:  # pragma: no cover - optional dependency
    paramiko = None  # type: ignore[assignment]


SUPPORTED_MODELS = ("dmpnn", "transformer", "chemberta")
DEFAULT_REPO_URL = "https://github.com/vstipetic/molecule-solubility-prediction.git"
DEFAULT_DOWNLOAD_DIR = "./runpod_downloads"
DEFAULT_SPLITS_DIR = "Storage/Datasets/splits"
REMOTE_SPLITS_DIR = "/workspace/data/splits"


def _project_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_splits_dir() -> Path:
    return _project_root() / DEFAULT_SPLITS_DIR


def _validate_splits_dir(path: Path) -> None:
    """Ensure a directory contains the required train/val/test CSV files."""
    if not path.is_dir():
        raise FileNotFoundError(
            f"Expected a splits directory at {path} "
            f"(containing train.csv, val.csv, test.csv)."
        )
    missing = [
        name for name in ("train.csv", "val.csv", "test.csv")
        if not (path / name).is_file()
    ]
    if missing:
        raise FileNotFoundError(
            f"Missing {', '.join(missing)} in {path}. "
            f"Run: python -m DataUtils.prepare_data --input <dataset.csv> "
            f"--output-dir {path}"
        )


# ---------------------------------------------------------------------------
# SSH helpers
# ---------------------------------------------------------------------------


def _load_ssh_key(path: Path):
    """Load an unencrypted private key, trying common key types in order."""
    if paramiko is None:
        raise RuntimeError(
            "paramiko is required for SSH/SCP. Install with `uv sync --extra runpod`."
        )
    errors = []
    for loader_name, loader in (
        ("ED25519", paramiko.Ed25519Key),
        ("RSA", paramiko.RSAKey),
        ("ECDSA", paramiko.ECDSAKey),
    ):
        try:
            return loader.from_private_key_file(str(path))
        except paramiko.SSHException as exc:  # noqa: PERF203
            errors.append(f"{loader_name}: {exc}")
    raise RuntimeError(
        f"Could not load SSH private key from {path}. Tried: {errors}"
    )


def _connect_ssh(host: str, port: int, key_path: Path) -> "paramiko.SSHClient":
    if paramiko is None:
        raise RuntimeError(
            "paramiko is required for SSH/SCP. Install with `uv sync --extra runpod`."
        )
    key = _load_ssh_key(key_path)
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    print(f"[ssh] connecting to root@{host}:{port} with {key_path}")
    client.connect(
        hostname=host, port=port, username="root", pkey=key, timeout=30
    )
    return client


def _run_command(ssh: "paramiko.SSHClient", command: str) -> int:
    """Run a command over SSH, streaming combined output to the console."""
    transport = ssh.get_transport()
    assert transport is not None
    chan = transport.open_session()
    chan.get_pty()
    chan.exec_command(command)
    while not chan.exit_status_ready():
        if chan.recv_ready():
            sys.stdout.write(chan.recv(4096).decode(errors="replace"))
            sys.stdout.flush()
        if chan.recv_stderr_ready():
            sys.stderr.write(chan.recv_stderr(4096).decode(errors="replace"))
            sys.stderr.flush()
        time.sleep(0.1)
    # Drain any remaining buffered output.
    while chan.recv_ready():
        sys.stdout.write(chan.recv(4096).decode(errors="replace"))
        sys.stdout.flush()
    while chan.recv_stderr_ready():
        sys.stderr.write(chan.recv_stderr(4096).decode(errors="replace"))
        sys.stderr.flush()
    return chan.recv_exit_status()


def _sftp_makedirs(sftp, remote_dir: str) -> None:
    """Recursively create a remote directory via SFTP."""
    parts = remote_dir.strip("/").split("/")
    cur = ""
    for part in parts:
        cur = f"{cur}/{part}"
        try:
            sftp.stat(cur)
        except FileNotFoundError:
            sftp.mkdir(cur)


def _sftp_upload(sftp, local: str, remote: str) -> None:
    """Upload a local file or directory tree to a remote path."""
    if os.path.isfile(local):
        _sftp_makedirs(sftp, os.path.dirname(remote))
        sftp.put(local, remote)
        print(f"[scp] uploaded file {local} -> {remote}")
        return
    # Directory: mirror recursively.
    _sftp_makedirs(sftp, remote)
    for entry in os.listdir(local):
        _sftp_upload(sftp, os.path.join(local, entry), f"{remote}/{entry}")


def _sftp_download_dir(sftp, remote_dir: str, local_dir: Path) -> None:
    """Recursively download a remote directory into a local directory."""
    local_dir.mkdir(parents=True, exist_ok=True)
    from stat import S_ISDIR
    for entry in sftp.listdir_attr(remote_dir):
        remote_path = f"{remote_dir}/{entry.filename}"
        local_path = local_dir / entry.filename
        if S_ISDIR(entry.st_mode):
            _sftp_download_dir(sftp, remote_path, local_path)
        else:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            sftp.get(remote_path, str(local_path))
            print(f"[scp] downloaded {remote_path} -> {local_path}")


def _sftp_download_file(sftp, remote_path: str, local_path: Path) -> bool:
    """Download a single remote file; return False if it does not exist."""
    try:
        sftp.stat(remote_path)
    except FileNotFoundError:
        return False
    local_path.parent.mkdir(parents=True, exist_ok=True)
    sftp.get(remote_path, str(local_path))
    print(f"[scp] downloaded {remote_path} -> {local_path}")
    return True


# ---------------------------------------------------------------------------
# Pod env construction
# ---------------------------------------------------------------------------


def _build_pod_env(args: argparse.Namespace, remote_data_path: str) -> List[dict]:
    """Build the env var list injected into the pod at creation time."""
    env: List[dict] = [
        {"key": "REPO_URL", "value": args.repo_url},
        {"key": "REPO_REF", "value": args.repo_ref},
        {"key": "MODEL", "value": args.model},
        {"key": "DATA_PATH", "value": remote_data_path},
        {"key": "CHECKPOINT_DIR", "value": "/workspace/checkpoints"},
        {"key": "TRAIN_ARGS", "value": args.training_args or ""},
    ]
    for var in ("WANDB_PROJECT", "WANDB_ENTITY", "WANDB_NAME", "WANDB_GROUP",
                "WANDB_TAGS", "WANDB_MODE", "WANDB_API_KEY"):
        value = os.environ.get(var)
        if value:
            env.append({"key": var, "value": value})
    return env


def _resolve_ssh_key(explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.environ.get("RUNPOD_SSH_PRIVATE_KEY")
    if env_path:
        return Path(env_path).expanduser()
    home = Path.home()
    for candidate in (".ssh/id_ed25519", ".ssh/id_rsa", ".ssh/id_ecdsa"):
        path = home / candidate
        if path.exists():
            return path
    raise RuntimeError(
        "No SSH private key found. Add your public key to RunPod -> Settings -> "
        "SSH Keys, set RUNPOD_SSH_PRIVATE_KEY in .env, or pass --ssh-private-key."
    )


def _parse_upload(spec: str) -> Tuple[str, str]:
    if ":" not in spec:
        raise argparse.ArgumentTypeError(
            f"--upload must be LOCAL:REMOTE, got {spec!r}"
        )
    local, remote = spec.split(":", 1)
    if not local or not remote:
        raise argparse.ArgumentTypeError(f"--upload must be LOCAL:REMOTE, got {spec!r}")
    return local, remote


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Train a model on a transient RunPod GPU pod."
    )
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS,
                        help="Which model to train (RF runs locally, not here)")
    parser.add_argument("--data-path", default=None,
                        help=f"Local splits directory with train/val/test CSVs "
                             f"(default: {DEFAULT_SPLITS_DIR})")
    parser.add_argument("--training-args", default="",
                        help='Extra args forwarded to the training module, e.g. '
                             '"--n-epochs 50 --batch-size 64"')
    parser.add_argument("--upload", action="append", type=_parse_upload, default=[],
                        help="Extra LOCAL:REMOTE file/dir to upload before training "
                             "(e.g. a pretrained checkpoint)")
    parser.add_argument("--gpu-type", default=DEFAULT_GPU_TYPE,
                        help=f"RunPod GPU type id (default: {DEFAULT_GPU_TYPE})")
    parser.add_argument("--image", default=DEFAULT_IMAGE,
                        help="Docker image (must include sshd)")
    parser.add_argument("--container-disk", type=int, default=50)
    parser.add_argument("--volume", type=int, default=50,
                        help="Persistent network volume in GB (holds the checkpoint)")
    parser.add_argument("--pod-name", default=None,
                        help="Pod display name (default: derived from model)")
    parser.add_argument("--ssh-private-key", default=None,
                        help="Path to the SSH private key matching your RunPod key")
    parser.add_argument("--repo-url", default=DEFAULT_REPO_URL)
    parser.add_argument("--repo-ref", default="main")
    parser.add_argument("--download-dir", default=DEFAULT_DOWNLOAD_DIR,
                        help="Local dir to receive the checkpoint and train.log")
    parser.add_argument("--keep-pod", action="store_true",
                        help="Do NOT terminate the pod on exit (for debugging)")
    parser.add_argument("--no-wandb", action="store_true",
                        help="Disable wandb on the pod (sets WANDB_MODE=disabled)")

    args = parser.parse_args()

    # Load .env (gitignored) so RUNPOD_API_KEY / WANDB_API_KEY are available.
    load_dotenv()

    if not os.environ.get("RUNPOD_API_KEY"):
        print("ERROR: RUNPOD_API_KEY env var is required.", file=sys.stderr)
        return 2

    splits_dir = Path(args.data_path) if args.data_path else _default_splits_dir()
    try:
        _validate_splits_dir(splits_dir)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2

    remote_data_path = REMOTE_SPLITS_DIR

    if paramiko is None:
        print(
            "ERROR: paramiko is not installed. Run `uv sync --extra runpod`.",
            file=sys.stderr,
        )
        return 2

    ssh_key = _resolve_ssh_key(args.ssh_private_key)
    if not ssh_key.exists():
        print(f"ERROR: SSH private key not found: {ssh_key}", file=sys.stderr)
        return 2

    if not os.environ.get("WANDB_API_KEY"):
        print("[runpod] WARNING: WANDB_API_KEY not set; pod will log to wandb "
              "in offline mode. Set WANDB_API_KEY to stream remotely.",
              file=sys.stderr)

    if args.no_wandb:
        os.environ["WANDB_MODE"] = "disabled"

    pod_env = _build_pod_env(args, remote_data_path)
    pod_name = args.pod_name or f"mol-sol-{args.model}"

    manager = PodManager()
    pod_id: Optional[str] = None

    try:
        print(f"[runpod] deploying pod {pod_name} (gpu={args.gpu_type}, "
              f"image={args.image})")
        pod_id = manager.create_pod(
            name=pod_name,
            image=args.image,
            gpu_type_id=args.gpu_type,
            container_disk_in_gb=args.container_disk,
            volume_in_gb=args.volume,
            env=pod_env,
        )
        print(f"[runpod] pod id: {pod_id}")

        pod = manager.wait_for_running(pod_id)
        host, port = manager.get_ssh_endpoint(pod)
        print(f"[runpod] pod running; ssh root@{host} -p {port}")

        ssh = _connect_ssh(host, port, ssh_key)
        try:
            sftp = ssh.open_sftp()
            try:
                # Upload bootstrap script.
                bootstrap_local = Path(__file__).parent / "bootstrap.sh"
                _sftp_upload(sftp, str(bootstrap_local), "/workspace/bootstrap.sh")

                # Upload pre-built train/val/test splits (entire directory).
                print(f"[scp] uploading splits {splits_dir} -> {remote_data_path}")
                _sftp_upload(sftp, str(splits_dir), remote_data_path)

                # Upload any extra artifacts (e.g. pretrained checkpoints).
                for local, remote in args.upload:
                    _sftp_upload(sftp, local, remote)
            finally:
                sftp.close()

            # Run the bootstrap script; stream output live.
            print("[runpod] starting training on pod...")
            exit_code = _run_command(ssh, "bash /workspace/bootstrap.sh")
            print(f"[runpod] bootstrap exited with code {exit_code}")

            # Fetch checkpoint(s) + log regardless of exit code (best effort).
            download_dir = Path(args.download_dir) / pod_id
            sftp = ssh.open_sftp()
            try:
                _sftp_download_dir(
                    sftp, "/workspace/checkpoints", download_dir / "checkpoints"
                )
                _sftp_download_file(
                    sftp, "/workspace/train.log", download_dir / "train.log"
                )
            finally:
                sftp.close()

            if exit_code != 0:
                print(f"[runpod] training failed (exit {exit_code}); "
                      f"see {download_dir}/train.log", file=sys.stderr)
                return exit_code
        finally:
            ssh.close()

        print(f"\n[runpod] done. Checkpoint + log in {download_dir}")
        print(f"[runpod] wandb project: "
              f"{os.environ.get('WANDB_PROJECT', 'mol-solubility')} "
              f"(entity: {os.environ.get('WANDB_ENTITY', '<default>')})")
        return 0

    finally:
        if pod_id is not None and not args.keep_pod:
            print(f"[runpod] terminating pod {pod_id}...")
            try:
                manager.terminate_pod(pod_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[runpod] WARNING: terminate failed: {exc}", file=sys.stderr)
        elif pod_id is not None:
            print(f"[runpod] --keep-pod set; pod {pod_id} left running.")


if __name__ == "__main__":
    sys.exit(main())
