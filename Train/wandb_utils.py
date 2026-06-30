"""Shared Weights & Biases logging helpers for training and validation scripts.

Centralizes ``wandb.init``/``wandb.finish`` so every training entry point
behaves consistently, including on headless RunPod pods where an interactive
wandb login prompt would otherwise hang the process.

Environment variables respected (all optional):
    WANDB_API_KEY:  API key for remote logging. When unset and ``mode`` is not
        explicitly forced, the run is started in ``offline`` mode so training
        never blocks on a prompt. Set ``WANDB_MODE=online`` to force remote.
    WANDB_ENTITY:   Default entity (account/team) for runs.
    WANDB_PROJECT:  Default project name.
    WANDB_DIR:      Local directory wandb writes run metadata to.
    WANDB_MODE:     Forces wandb mode (e.g. ``online``, ``offline``, ``disabled``).

Robustness notes:
    - Every metric-logging call site in the codebase is guarded by
      ``wandb.run is not None``, so calling :func:`init_run` with
      ``mode="disabled"`` (or ``--no-wandb``) degrades gracefully.
    - On RunPod, set ``WANDB_API_KEY`` (and optionally ``WANDB_ENTITY``) as pod
      env vars; the pod reaches wandb over standard HTTPS egress.
"""

import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Union

import wandb

DEFAULT_PROJECT = "mol-solubility"


def load_env_file(path: Union[str, Path]) -> bool:
    """Load ``KEY=VALUE`` lines from a file into ``os.environ``.

    Existing environment variables are NOT overridden, so real shell exports
    take precedence over the file. Blank lines and ``#`` comments are skipped;
    surrounding quotes around values are stripped.

    Args:
        path: Path to an env file (e.g. ``.env``).

    Returns:
        True if the file was loaded, False if it did not exist.
    """
    p = Path(path)
    if not p.is_file():
        return False
    for raw in p.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value
    return True


def load_dotenv(extra_paths: Optional[Sequence[Union[str, Path]]] = None) -> None:
    """Load ``.env`` from the cwd and the project root, if present.

    Safe to call multiple times; existing env vars always win. This lets
    ``.env`` hold ``RUNPOD_API_KEY`` / ``WANDB_API_KEY`` etc. without committing
    secrets (``.env`` is gitignored).
    """
    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parent.parent / ".env",
    ]
    if extra_paths:
        candidates.extend(Path(p) for p in extra_paths)
    for path in candidates:
        load_env_file(path)


def _parse_tags(tags: Optional[Union[str, Sequence[str]]]) -> Optional[List[str]]:
    """Normalize a tag argument into a list of strings.

    Accepts a comma/space separated string or a sequence. Returns ``None`` when
    no tags are provided so wandb keeps its default behavior.
    """
    if tags is None:
        return None
    if isinstance(tags, str):
        parsed = [t.strip() for t in tags.replace(",", " ").split() if t.strip()]
        return parsed or None
    parsed = [str(t) for t in tags if t]
    return parsed or None


def init_run(
    project: str = DEFAULT_PROJECT,
    name: Optional[str] = None,
    config: Optional[Dict] = None,
    job_type: str = "train",
    tags: Optional[Union[str, Sequence[str]]] = None,
    group: Optional[str] = None,
    entity: Optional[str] = None,
    mode: Optional[str] = None,
    dir: Optional[str] = None,
    run_id: Optional[str] = None,
    resume: Optional[Union[str, bool]] = None,
) -> "wandb.sdk.wandb_run.Run":
    """Initialize a wandb run with robust, environment-aware defaults.

    Args:
        project: wandb project name. Falls back to ``WANDB_PROJECT`` env var,
            then :data:`DEFAULT_PROJECT`.
        name: Human-readable run name.
        config: Hyperparameter/config dict to log.
        job_type: wandb job type (e.g. ``"train"``, ``"finetune"``, ``"pretrain"``).
        tags: Run tags. Accepts a string (comma/space separated) or a sequence.
        group: wandb group name (useful to group RunPod runs by model).
        entity: wandb entity. Falls back to ``WANDB_ENTITY`` env var.
        mode: Force wandb mode. Falls back to ``WANDB_MODE`` env var, then to
            ``"offline"`` when no ``WANDB_API_KEY`` is set.
        dir: Local run directory. Falls back to ``WANDB_DIR`` env var, then
            ``./wandb``.
        run_id: Optional stable run id for resume.
        resume: Resume policy (``"allow"``, ``"must"``, ``"never"``, or bool).

    Returns:
        The active ``wandb.Run``.
    """
    load_dotenv()

    if not project:
        project = os.environ.get("WANDB_PROJECT", DEFAULT_PROJECT)

    if entity is None:
        entity = os.environ.get("WANDB_ENTITY")

    if mode is None:
        mode = os.environ.get("WANDB_MODE")
    if mode is None:
        if not os.environ.get("WANDB_API_KEY"):
            print(
                "[wandb] WANDB_API_KEY not set; starting in offline mode. "
                "Set WANDB_MODE=online or provide WANDB_API_KEY to log remotely."
            )
            mode = "offline"

    if dir is None:
        dir = os.environ.get("WANDB_DIR", "./wandb")

    return wandb.init(
        project=project,
        name=name,
        config=config,
        job_type=job_type,
        tags=_parse_tags(tags),
        group=group,
        entity=entity,
        mode=mode,
        dir=dir,
        id=run_id,
        resume=resume,
    )


def finish_run() -> None:
    """Finish the active wandb run if one exists."""
    if wandb.run is not None:
        wandb.finish()


def is_active() -> bool:
    """Return whether a wandb run is currently active."""
    return wandb.run is not None
