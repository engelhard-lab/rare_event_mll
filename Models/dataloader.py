import numpy as np


def create_batches(num_samples, batch_size, num_epochs, random_seed):
    epochs = []
    rs = np.random.RandomState(random_seed)
    for e in range(num_epochs):
        this_batch = []
        order = np.arange(num_samples)
        rs.shuffle(order)
        i = 0
        while i + batch_size < num_samples:
            this_batch.append(np.copy(order[i:i+batch_size]))
            i += batch_size
        epochs.append(this_batch)
    return epochs
