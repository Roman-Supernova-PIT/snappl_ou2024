def test_version_is_string():
    from snappl_ou2024 import __version__
    assert isinstance(__version__, str)
