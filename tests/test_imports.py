def test_core_imports():
    import numpy  # noqa: F401
    import scipy  # noqa: F401
    import pandas  # noqa: F401
    import polars  # noqa: F401
    import pyarrow  # noqa: F401
    import sklearn  # noqa: F401


def test_prisma_modules_import():
    import builder  # noqa: F401
    import loader  # noqa: F401
    import partition  # noqa: F401
    import qc  # noqa: F401
    import solver  # noqa: F401
    import tune_rank  # noqa: F401
