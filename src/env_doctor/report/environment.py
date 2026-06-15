"""Runtime environment detection for choosing an output renderer."""


def is_notebook() -> bool:
    """Return True when running inside a Jupyter/IPython notebook kernel.

    Detects the ZMQ-based interactive shell used by Jupyter (notebook,
    JupyterLab, VS Code, Colab). Returns False for plain terminals, the
    classic IPython REPL, and when IPython is not installed at all.

    IPython is imported lazily so it never becomes a hard dependency.
    """
    try:
        from IPython import get_ipython

        shell = get_ipython().__class__.__name__
    except (ImportError, AttributeError):
        return False

    # ZMQInteractiveShell -> Jupyter; Google Colab uses its own subclass name.
    return shell in ("ZMQInteractiveShell", "Shell")
