import logging

import pandas as pd

logger = logging.getLogger(__name__)


def find_duplicates(
    df: pd.DataFrame, name: str, key: str, strict: bool = False
) -> pd.DataFrame:
    """Find duplicates key in dataframe and eventually drop them

    :param df: _description_
    :type df: pd.DataFrame
    :param name: _description_
    :type name: str
    :param key: _description_
    :type key: str
    :param strict: _description_, defaults to False
    :type strict: bool, optional
    :raises RuntimeError: _description_
    :return: _description_
    :rtype: _type_
    """

    counts = df[key].value_counts()
    if (counts > 1).any():
        total_duplicates = (counts[counts > 1] - 1).sum()
        logger.warning(
            f"Duplicates found in {key}: {name} — {total_duplicates} duplicate(s) removed"
        )
        logger.debug(f"Duplicate counts for {key} in {name}:\n{counts[counts > 1]}")
        if strict:
            raise RuntimeError(f"Strict mode: duplicates found in {key} for {name}")
    df_clean = df.drop_duplicates(subset=[key])
    return df_clean
