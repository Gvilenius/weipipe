import torch
from torch.utils.data import Dataset


class MyData(Dataset):
    def __init__(self, n):
        torch.manual_seed(1234)
        n_samples = 10
        seq_len = n
        input = torch.randint(0, 10, (n_samples, seq_len + 1))
        enc_input = input[:, :-1].long()
        targets = input[:, 1:].long()
        self.data = list(zip(enc_input, targets))

    def print(self):
        import pprint

        pprint.pprint(self.data)

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)
