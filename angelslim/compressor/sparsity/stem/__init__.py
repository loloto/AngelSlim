"""Stem — Sparse Token Estimation Module for long-context LLM inference.

Public API:
    StemInference: Callable that patches a HuggingFace model to use Stem
        sparse attention during the prefill stage.
"""

from .models_patch import StemInference

__all__ = ["StemInference"]
