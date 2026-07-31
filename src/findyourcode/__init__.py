from importlib.metadata import PackageNotFoundError, version

try:  # the version lives in pyproject.toml and nowhere else
    __version__ = version("findyourcode")
except PackageNotFoundError:  # running from a source tree that was never installed
    __version__ = "0.0.0+unknown"

__all__ = ["__version__"]
