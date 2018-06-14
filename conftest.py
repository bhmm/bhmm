import pytest

@pytest.fixture(autouse=True)
def add_np(doctest_namespace):
    # we enforce legacy string formatting of numpy arrays, because the output format changed in version 1.14,
    # leading to failing doctests.
    import numpy as np
    try:
        np.set_printoptions(legacy='1.13')
    except TypeError:
        pass


def pytest_collection_modifyitems(session, config, items):
    for i in items:
        if '_external' in i.location[0]:
            i.add_marker(pytest.mark.skip('external'))
