"""Causal steering of tool-feedback integration in LLM agents.

Модули без torch (импортируются всегда):
    metrics, simulator, probe

Модули, которым нужен torch/transformers (импортируются явно):
    steering, windows

    from steering_tool_feedback import metrics, simulator
    from steering_tool_feedback.windows import prepare_prompt, describe_windows
"""

from . import metrics, probe, simulator

__all__ = ["metrics", "probe", "simulator"]
__version__ = "0.1.0"
