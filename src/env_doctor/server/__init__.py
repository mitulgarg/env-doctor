"""Dashboard server package for env-doctor fleet monitoring."""


def start_server(host: str = "0.0.0.0", port: int = 8765):
    """Start the dashboard server."""
    import uvicorn
    from .app import app
    uvicorn.run(app, host=host, port=port, log_level="info")
