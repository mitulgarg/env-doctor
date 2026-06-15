"""Public Python API for env-doctor, designed for notebook use.

Typical usage inside a Jupyter notebook::

    from env_doctor import check
    check()          # last cell expression -> rich HTML report

In a plain terminal / script the same call prints the familiar text report.
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from .report import is_notebook, format_result_html


class CheckReport:
    """Result of :func:`check`, with format-aware display hooks.

    - In Jupyter, being the last expression in a cell triggers
      ``_repr_html_`` and renders the rich HTML report.
    - In an interactive terminal, ``__repr__`` shows a one-line summary
      (the full text report is printed by :func:`check` itself).
    - ``.html`` and ``.to_dict()`` expose the report for programmatic use.
    """

    def __init__(self, output: Dict[str, Any]) -> None:
        self._output = output

    def to_dict(self) -> Dict[str, Any]:
        """Return the underlying structured result dict."""
        return self._output

    @property
    def html(self) -> str:
        """Return the self-contained HTML report as a string."""
        return format_result_html(self._output)

    def _repr_html_(self) -> str:  # noqa: D401 - IPython display protocol
        return self.html

    def __repr__(self) -> str:
        summary = self._output.get("summary", {})
        return (
            f"<CheckReport status={self._output.get('status', 'unknown')} "
            f"issues={summary.get('issues_count', 0)}>"
        )


def check(format: Optional[str] = None) -> CheckReport:
    """Run the environment check and render it for the current environment.

    In a notebook, return the result as a cell's last expression and Jupyter
    renders the rich HTML automatically (via ``_repr_html_``) — no explicit
    ``display()`` call, so it never double-renders. In a terminal/script the
    familiar text report is printed.

    Args:
        format: Force a renderer regardless of environment: ``"text"`` always
            prints the terminal report; ``"html"`` skips the text print and
            leaves rendering to the returned object's ``_repr_html_`` (use
            ``check(format="html").html`` for the raw string). ``None``
            (default) auto-detects: HTML in a notebook, text in a terminal.

    Returns:
        A :class:`CheckReport`.
    """
    # Imported here to avoid a circular import (cli imports report, api).
    from .cli import collect_check_results, render_check_text

    bundle = collect_check_results()
    report = CheckReport(bundle["output"])

    render_html = format == "html" or (format is None and is_notebook())
    if not render_html:
        render_check_text(bundle)

    return report