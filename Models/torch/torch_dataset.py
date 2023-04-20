import numpy as np
import torch


class Dataset(torch.utils.data.Dataset):
    def __init__(self, xx, y1=None, y2=None):
        self.X = xx.astype('float32')
        if y2 is not None:
            self.y = np.concatenate([y1.astype('float32').reshape(-1, 1),
                                     y2.astype('float32').reshape(-1, 1)],
                                    axis=1)
        elif y1 is not None:
            self.y = y1.astype('float32').reshape(-1, 1)
        else:  # no labels, used for estimation set
            self.y = np.zeros(shape=(xx.shape[0], 1)).astype('float32')

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx, :]
