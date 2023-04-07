def flatten(S: list) -> list:
    """
    flatten list of list
    """
    from pandas.core.common import flatten as pd_flatten

    return list(pd_flatten(S))
