"""Rendering helpers for env-doctor check results.

This package keeps presentation concerns (HTML rendering, runtime
environment detection) decoupled from the detection logic in ``cli.py``.
"""
from .environment import is_notebook
from .html import format_result_html

__all__ = ["is_notebook", "format_result_html"]
