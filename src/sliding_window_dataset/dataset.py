from typing import Callable, Optional, Sequence

from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """A dataset that iterates over time series data using a sliding window."""

    def __init__(
        self,
        *data: Sequence,
        window_size: int,
        transform: Optional[Callable[[Sequence], Sequence]] = None,
    ):
        if not data:
            raise ValueError("At least one time series must be provided.")

        if not all(len(series) == len(data[0]) for series in data):
            raise ValueError("All time series must have the same length.")

        if len(data[0]) < window_size:
            raise ValueError("Time series cannot be smaller than window.")

        self.data = data
        self.window_size = window_size
        self.transform = transform
        self.n_windows = len(data[0]) - window_size + 1

    def __len__(self) -> int:
        return len(self.data) * self.n_windows

    def __getitem__(self, idx: int) -> Sequence:
        series, timestep = divmod(idx, self.n_windows)
        window = self.data[series][timestep : timestep + self.window_size]
        return self.transform(window) if self.transform else window
