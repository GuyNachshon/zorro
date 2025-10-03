import pytest
from unittest.mock import MagicMock
from typing import List
import matplotlib.patches as m_patches
from cbsplotlib.utils import format_thousands_label, swap_legend_boxes


def test_format_thousands_label_positive_integers():
    """
    Test format_thousands_label with positive integers.

    This test verifies that positive integers are correctly formatted
    with spaces as thousand separators.
    """
    assert format_thousands_label(1234, None) == "1 234"
    assert format_thousands_label(1234567, None) == "1 234 567"


def test_format_thousands_label_negative_integers():
    """
    Test format_thousands_label with negative integers.

    This test verifies that negative integers are correctly formatted
    with spaces as thousand separators and a negative sign.
    """
    assert format_thousands_label(-1234, None) == "-1 234"
    assert format_thousands_label(-1234567, None) == "-1 234 567"
    assert format_thousands_label(-1234, None) == "-1 234"
    assert format_thousands_label(-1234567, None) == "-1 234 567"


def test_format_thousands_label_floating_point_numbers():
    """
    Test format_thousands_label with floating point numbers.

    This test verifies that floating point numbers are correctly formatted
    with spaces as thousand separators and the decimal part is removed.
    """
    assert format_thousands_label(1234.56, None) == "1 234"
    assert format_thousands_label(1234567.89, None) == "1 234 567"
    assert format_thousands_label(1234.56, None) == "1 234"
    assert format_thousands_label(1234567.89, None) == "1 234 567"


def test_format_thousands_label_zero():
    """
    Test format_thousands_label with zero.

    This test verifies that zero is correctly formatted
    as a string "0" without any thousand separators.
    """
    assert format_thousands_label(0, None) == "0"


def test_format_thousands_label_non_numeric_input():
    """
    Test format_thousands_label with non-numeric input.

    This test verifies that passing a non-numeric input to
    format_thousands_label raises a TypeError.
    """
    with pytest.raises(ValueError):
        format_thousands_label("abc", None)


def test_single_column():
    """
    Test `swap_legend_boxes` with a single column.

    This test checks that when the number of columns (`n_cols`) is set to 1,
    the function `swap_legend_boxes` returns the handles and labels in the
    same order as they were provided.
    """
    handles = [MagicMock(m_patches.Patch) for _ in range(5)]
    labels = ["label1", "label2", "label3", "label4", "label5"]
    n_cols = 1
    reordered_handles, reordered_labels = swap_legend_boxes(handles, labels, n_cols)
    assert reordered_handles == handles
    assert reordered_labels == labels


def test_multiple_columns():
    """
    Test `swap_legend_boxes` with multiple columns.

    This test checks that when the number of columns (`n_cols`) is greater than 1,
    the function `swap_legend_boxes` correctly rearranges the handles and labels
    in the order of the first row.

    """
    handles = [MagicMock(m_patches.Patch) for _ in range(6)]
    labels = ["label1", "label2", "label3", "label4", "label5", "label6"]
    n_cols = 2
    reordered_handles, reordered_labels = swap_legend_boxes(handles, labels, n_cols)
    assert reordered_handles == [
        handles[0],
        handles[2],
        handles[4],
        handles[1],
        handles[3],
        handles[5],
    ]
    assert reordered_labels == [
        "label1",
        "label3",
        "label5",
        "label2",
        "label4",
        "label6",
    ]


def test_empty_lists():
    """
    Test `swap_legend_boxes` with empty lists.

    This test verifies that when both `handles` and `labels` are empty lists,
    the function `swap_legend_boxes` returns empty lists for both reordered
    handles and labels, regardless of the value of `n_cols`.
    """
    handles = []
    labels = []
    n_cols = 1
    reordered_handles, reordered_labels = swap_legend_boxes(handles, labels, n_cols)
    assert reordered_handles == []
    assert reordered_labels == []


def test_single_handle_and_label():
    """
    Test `swap_legend_boxes` with single handle and label.

    This test checks that when there is only one handle and label, the function
    `swap_legend_boxes` simply returns the original handle and label, regardless
    of the value of `n_cols`.
    """
    handles = [MagicMock(m_patches.Patch)]
    labels = ["label1"]
    n_cols = 1
    reordered_handles, reordered_labels = swap_legend_boxes(handles, labels, n_cols)
    assert reordered_handles == handles
    assert reordered_labels == labels
    handles = [MagicMock(m_patches.Patch)]
    labels = ["label1"]
    n_cols = 1
    reordered_handles, reordered_labels = swap_legend_boxes(handles, labels, n_cols)
    assert reordered_handles == handles
    assert reordered_labels == labels


def test_mismatch_between_handles_and_labels():
    """
    Test `swap_legend_boxes` with mismatched handles and labels.

    This test verifies that when the number of handles and labels do not match,
    the function `swap_legend_boxes` raises a ValueError.
    """
    handles = [MagicMock(m_patches.Patch) for _ in range(5)]
    labels = ["label1", "label2", "label3"]
    n_cols = 1
    with pytest.raises(ValueError):
        swap_legend_boxes(handles, labels, n_cols)
