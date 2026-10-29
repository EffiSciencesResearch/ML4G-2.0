from meta.notebook_tests.notebooks import list_problems


def test_every_workshop_notebook_is_listed():
    assert list_problems() == []
