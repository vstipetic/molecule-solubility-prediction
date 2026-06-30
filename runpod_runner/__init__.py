"""RunPod training orchestration package.

Spins up a single transient GPU pod, trains one of the project's models, and
tears the pod down. Named ``runpod_runner`` (not ``runpod``) to avoid shadowing
the official ``runpod`` PyPI SDK that :mod:`runpod_runner.pod_manager` imports.

See :mod:`runpod_runner.train_on_runpod` for the orchestrator and
:mod:`runpod_runner.pod_manager` for the pod lifecycle wrapper.
"""

from runpod_runner.pod_manager import PodManager

__all__ = ["PodManager"]
