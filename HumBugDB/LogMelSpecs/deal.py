import numpy as np
import torch
import torch.nn as nn
from torch.nn import Embedding, LSTM
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence


class MinimalDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def padding(data, id):
    DATA = data
    DATA = list(map(lambda x: torch.tensor(x), DATA))
    # 词典大小，包含了padding token 0
    print(id)
    NUM_WORDS = 10
    BATCH_SIZE = id
    LSTM_DIM = 5  # hidden dim

    dataset = MinimalDataset(DATA)
    data_loader = DataLoader(dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             collate_fn=lambda x: x)

    # print(next(iter(data_loader)))

    # iterate through the dataset:
    # for i, batch in enumerate(data_loader):
        # print(f'{i}, {batch}')

    # 输出：
    # 0, [tensor([1, 2, 3]), tensor([4, 5]), tensor([6, 7, 8, 9])]
    # 1, [tensor([4, 6, 2, 9, 0])]

    # this always gets you the first batch of the dataset:
    batch = next(iter(data_loader))
    padded = pad_sequence(batch, batch_first=True)
    # print(f' [0] padded: \n{padded}\n')
    # print("--------------")
    print(padded)
    return padded
