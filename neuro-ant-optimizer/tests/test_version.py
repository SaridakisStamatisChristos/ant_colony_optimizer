def test_version_string_present():
    import neuro_ant_optimizer as nao

    assert isinstance(nao.__version__, str)
    assert len(nao.__version__) > 0
