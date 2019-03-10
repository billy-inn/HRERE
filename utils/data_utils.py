import numpy as np

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((data_size - 1) / batch_size) + 1
    for epoch in range(num_epochs):
        if shuffle:
            np.random.seed(2019)
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def flatten(words_batch, positions_batch, labels_batch):
    words = []
    positions = []
    labels = []
    shapes = []
    cnt = 0
    for i in range(len(labels_batch)):
        shapes.append(cnt)
        labels.append(labels_batch[i][0])
        cnt += len(labels_batch[i])
        for word in words_batch[i]:
            words.append(word)
        for pos in positions_batch[i]:
            positions.append(pos)
    shapes.append(cnt)
    words = np.array(words)
    positions = np.array(positions)
    shapes = np.array(shapes)
    labels = np.array(labels)
    return words, positions, shapes, labels

def load_dict_from_txt(path):
    d = {}
    with open(path) as f:
        for line in f.readlines():
            a, b = line.strip().split()
            d[a] = int(b)
    return d
