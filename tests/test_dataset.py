import pytest

from sliding_window_dataset import SlidingWindowDataset


def test_no_data_raises():
    with pytest.raises(ValueError):
        SlidingWindowDataset(window_size=3)


def test_different_length_data_raises():
    series1 = [1, 2, 3, 4]
    series2 = [10, 20, 30]
    with pytest.raises(ValueError):
        SlidingWindowDataset(series1, series2, window_size=2)


def test_data_too_short_raises():
    series = [1, 2]
    with pytest.raises(ValueError):
        SlidingWindowDataset(series, window_size=3)


def test_length_single_series():
    series = list(range(10))
    window_size = 3
    dataset = SlidingWindowDataset(series, window_size=window_size)
    expected_n_windows = len(series) - window_size + 1
    assert len(dataset) == expected_n_windows


def test_length_multiple_series():
    series1 = list(range(10))
    series2 = list(range(10, 20))
    window_size = 3
    dataset = SlidingWindowDataset(series1, series2, window_size=window_size)
    expected_total = 2 * (len(series1) - window_size + 1)
    assert len(dataset) == expected_total


def test_getitem_without_transform():
    series = list(range(10))
    window_size = 3
    dataset = SlidingWindowDataset(series, window_size=window_size)
    last_index = len(series) - window_size
    assert dataset[0] == series[0:3]
    assert dataset[2] == series[2:5]
    assert dataset[last_index] == series[last_index : last_index + window_size]


def test_getitem_with_transform():
    series = list(range(10))
    transform = lambda window: sum(window)  # noqa: E731
    dataset = SlidingWindowDataset(series, window_size=3, transform=transform)
    assert dataset[0] == sum(series[0:3])
    assert dataset[2] == sum(series[2:5])


def test_getitem_multiple_series():
    series1 = list(range(10))
    series2 = list(range(10, 20))
    window_size = 3
    dataset = SlidingWindowDataset(series1, series2, window_size=window_size)
    n = len(series1) - window_size + 1

    # Windows from the first series
    assert dataset[0] == series1[0:3]
    assert dataset[n - 1] == series1[n - 1 : n - 1 + window_size]

    # Windows from the second series
    assert dataset[n] == series2[0:3]
    assert dataset[len(dataset) - 1] == series2[-window_size:]
