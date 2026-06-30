"""RunPod pod lifecycle management via the official ``runpod`` Python SDK.

Wraps pod create / poll / SSH-endpoint discovery / stop / terminate so the
orchestrator (:mod:`runpod_runner.train_on_runpod`) can spin up a single GPU
pod, run a training job over SSH, and tear it down reliably.

Prerequisites:
    - ``runpod`` package installed (``uv sync --extra runpod``).
    - ``RUNPOD_API_KEY`` env var set, or pass ``api_key``.
    - An SSH **public** key registered under your RunPod account settings so the
      pod auto-injects it into ``~/.ssh/authorized_keys`` at startup. The
      matching private key is used by the orchestrator for SSH/SCP.
"""

import os
import time
from typing import Any, Dict, List, Optional, Tuple

import runpod

DEFAULT_GPU_TYPE = "NVIDIA RTX A6000"
DEFAULT_IMAGE = "runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04"


def _attr(obj: Any, name: str, default: Any = None) -> Any:
    """Read a field from a dict or object, tolerating both shapes."""
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class PodManager:
    """Manage a single RunPod training pod."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        key = api_key or os.environ.get("RUNPOD_API_KEY")
        if not key:
            raise RuntimeError(
                "RUNPOD_API_KEY is not set. Get it from RunPod -> Settings -> "
                "API Keys and export it before running the orchestrator."
            )
        runpod.api_key = key

    def create_pod(
        self,
        name: str,
        image: str = DEFAULT_IMAGE,
        gpu_type_id: str = DEFAULT_GPU_TYPE,
        gpu_count: int = 1,
        container_disk_in_gb: int = 50,
        volume_in_gb: int = 50,
        volume_mount_path: str = "/workspace",
        ports: str = "22/tcp",
        env: Optional[List[Dict[str, str]]] = None,
        support_public_ip: bool = True,
        cloud_type: str = "ALL",
        min_vcpu_count: int = 4,
        min_memory_in_gb: int = 15,
    ) -> str:
        """Deploy an on-demand pod and return its id.

        Args:
            name: Pod display name.
            image: Docker image (must ship an SSH server for SCP).
            gpu_type_id: RunPod GPU type id (e.g. ``"NVIDIA RTX A6000"``).
            gpu_count: Number of GPUs.
            container_disk_in_gb: Container (ephemeral) disk size.
            volume_in_gb: Persistent network volume size (mounted at
                ``volume_mount_path``); checkpoints are written here so they
                survive a stop until terminate.
            volume_mount_path: Where the volume is mounted in the container.
            ports: Exposed ports; SSH on 22/tcp is required for the
                orchestrator to connect.
            env: List of ``{"key": ..., "value": ...}`` env vars injected into
                the pod (used for WANDB_* and run config).
            support_public_ip: Required to reach SSH from outside RunPod.
            cloud_type: ``"ALL"`` searches all clouds for availability.
            min_vcpu_count: Minimum vCPUs.
            min_memory_in_gb: Minimum host memory.

        Returns:
            The pod id.
        """
        pod = runpod.create_pod(
            name=name,
            image_name=image,
            gpu_type_id=gpu_type_id,
            cloud_type=cloud_type,
            gpu_count=gpu_count,
            container_disk_in_gb=container_disk_in_gb,
            volume_in_gb=volume_in_gb,
            volume_mount_path=volume_mount_path,
            ports=ports,
            env=env,
            support_public_ip=support_public_ip,
            min_vcpu_count=min_vcpu_count,
            min_memory_in_gb=min_memory_in_gb,
        )
        pod_id = _attr(pod, "id")
        if not pod_id:
            raise RuntimeError(f"Pod deployment did not return an id: {pod!r}")
        return str(pod_id)

    def get_pod(self, pod_id: str) -> Any:
        """Return the current pod object."""
        return runpod.get_pod(pod_id)

    def wait_for_running(
        self,
        pod_id: str,
        timeout: float = 900.0,
        poll_interval: float = 10.0,
    ) -> Any:
        """Poll until the pod is RUNNING and has a public SSH endpoint.

        Args:
            pod_id: Pod id.
            timeout: Maximum seconds to wait.
            poll_interval: Seconds between polls.

        Returns:
            The pod object once running.

        Raises:
            TimeoutError: If the pod does not become running in time.
        """
        deadline = time.time() + timeout
        last = None
        while time.time() < deadline:
            pod = self.get_pod(pod_id)
            last = pod
            status = _attr(pod, "desiredStatus")
            runtime = _attr(pod, "runtime")
            print(f"[runpod] pod {pod_id} status={status}")
            if status == "RUNNING" and runtime is not None:
                host, port = self._extract_ssh_endpoint(pod)
                if host and port:
                    return pod
            time.sleep(poll_interval)
        raise TimeoutError(
            f"Pod {pod_id} did not reach RUNNING with an SSH endpoint within "
            f"{timeout}s. Last state: {last!r}"
        )

    @staticmethod
    def _extract_ssh_endpoint(pod: Any) -> Tuple[Optional[str], Optional[int]]:
        """Best-effort extraction of (public_ip, ssh_port) from a pod object."""
        runtime = _attr(pod, "runtime")
        host = _attr(runtime, "publicIp") or _attr(runtime, "ingressIp")

        # Shape 1: portMappings {"22": 10341}
        port_mappings = _attr(runtime, "portMappings")
        if isinstance(port_mappings, dict) and "22" in port_mappings:
            return host, int(port_mappings["22"])

        # Shape 2: ports {"22/tcp": [{"hostPort": ...}]} or {"22/tcp": 10341}
        ports = _attr(runtime, "ports")
        if isinstance(ports, dict):
            entry = ports.get("22/tcp") or ports.get("22")
            if isinstance(entry, list) and entry:
                return host, int(_attr(entry[0], "hostPort", entry[0]))
            if entry is not None:
                return host, int(entry)

        return host, None

    def get_ssh_endpoint(self, pod: Any) -> Tuple[str, int]:
        """Return (public_ip, ssh_port), raising if unavailable."""
        host, port = self._extract_ssh_endpoint(pod)
        if not host or not port:
            raise RuntimeError(
                f"Could not determine SSH endpoint for pod. "
                f"Ensure ports='22/tcp' and a registered SSH key. Pod: {pod!r}"
            )
        return str(host), int(port)

    def stop_pod(self, pod_id: str) -> None:
        """Stop (pause) a pod without destroying the volume."""
        runpod.stop_pod(pod_id)

    def terminate_pod(self, pod_id: str) -> None:
        """Permanently terminate a pod and its volume."""
        runpod.terminate_pod(pod_id)
