from torch.utils.data import Dataset


class SlidingWindowDataset(Dataset):
    """A dataset that iterates over data using a sliding window."""

    def __init__(self, *data, window: int):
        if not data:
            raise ValueError("At least one iterable must be provided.")

        length = len(data[0])
        if any(len(d) != length for d in data):
            raise ValueError("All iterables must have the same length.")

        self.data = data
        self.window = window
        self.size = length - window

    def __getitem__(self, index):
        return tuple(d[index : index + self.window] for d in self.data)

    def __len__(self):
        return self.size
