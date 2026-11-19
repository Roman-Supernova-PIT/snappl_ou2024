from importlib.metadata import version, PackageNotFoundError
__all__ = []



try:
    __version__ = version("roman_snpit_snappl_ou2024")
except PackageNotFoundError:
    # package is not installed
    pass
