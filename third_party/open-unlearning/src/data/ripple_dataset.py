# ruff: noqa
import json

from torch.utils.data import Dataset


class RippleUnlearningDataset(Dataset):
    """
    Dataset for the Ripple Unlearning benchmark.
    Each item in the dataset corresponds to a single unlearning case,
    containing the fact to forget and all associated probes.
    """

    def __init__(self, path: str):
        self.path = path
        self.data = []
        with open(self.path, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

