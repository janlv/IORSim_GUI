# src/iorsim/version.py
try:
    from _version import version as __version__
except ImportError:
    __version__ = "0.0.0"  # fallback for dev mode or if not built yet
