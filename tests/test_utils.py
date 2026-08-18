"""Unit tests for python_magnetrun utils modules.

Covers:
  - list: flatten
  - duplicates: find_duplicates
"""

import pandas as pd

from python_magnetrun.utils.duplicates import find_duplicates
from python_magnetrun.utils.list import flatten


class TestFlatten:
    def test_nested_list(self) -> None:
        assert flatten([[1, 2], [3, 4]]) == [1, 2, 3, 4]

    def test_already_flat(self) -> None:
        assert flatten([1, 2, 3]) == [1, 2, 3]

    def test_empty(self) -> None:
        assert flatten([]) == []

    def test_deeply_nested(self) -> None:
        assert flatten([[1, [2, 3]], [4]]) == [1, 2, 3, 4]

    def test_string_elements(self) -> None:
        assert flatten([["a", "b"], ["c"]]) == ["a", "b", "c"]


class TestFindDuplicates:
    def test_no_duplicates_returns_same_length(self) -> None:
        df = pd.DataFrame({"id": [1, 2, 3], "val": [10, 20, 30]})
        result = find_duplicates(df, name="test", key="id")
        assert len(result) == 3

    def test_with_duplicates_drops_extras(self) -> None:
        df = pd.DataFrame({"id": [1, 1, 2, 3], "val": [10, 11, 20, 30]})
        result = find_duplicates(df, name="test", key="id")
        assert len(result) == 3

    def test_all_duplicates_keeps_one(self) -> None:
        df = pd.DataFrame({"id": [5, 5, 5], "val": [1, 2, 3]})
        result = find_duplicates(df, name="test", key="id")
        assert len(result) == 1

    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        result = find_duplicates(df, name="test", key="id")
        assert isinstance(result, pd.DataFrame)

    def test_columns_preserved(self) -> None:
        df = pd.DataFrame({"id": [1, 2], "val": [10, 20]})
        result = find_duplicates(df, name="test", key="id")
        assert list(result.columns) == ["id", "val"]
