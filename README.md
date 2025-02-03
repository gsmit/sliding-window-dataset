# Sliding Window Dataset
A minimal Python library for slicing time series data into overlapping sliding windows. Designed as a subclass of PyTorch’s `Dataset`, it seamlessly integrates with your data pipelines and model training routines. The dataset supports lists, data frames, arrays, tensors, and any other type of sequence-like data.

## Installation
Install by cloning this repository:
```bash
https://github.com/gsmit/sliding-window-dataset.git
cd sliding-window-dataset
pip install -e .
```

## Usage
To create a sliding window dataset, simply run the following:

```python
from sliding_window_dataset import SlidingWindowDataset

# Create two simple time series
series1 = list(range(5))
series2 = list(range(10, 15))
window_size = 3

# Initialize the dataset
dataset = SlidingWindowDataset(series1, series2, window_size=window_size)

# Print each sliding window
for window in dataset:
    print(window)
```

This will output:
```python
[0, 1, 2]
[1, 2, 3]
[2, 3, 4]
[10, 11, 12]
[11, 12, 13]
[12, 13, 14]
```
