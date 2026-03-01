"""Upscale orchestrator — ties model loading, video I/O, and inference together."""

from __future__ import annotations


class UpscaleEngine:
    """Orchestrates the full upscale pipeline.

    load config -> download model -> read video -> enforce constraints
    -> run inference -> write output
    """
