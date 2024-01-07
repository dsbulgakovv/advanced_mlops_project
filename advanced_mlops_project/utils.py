import os

import numpy as np
import torch
import torchvision
from matplotlib import pyplot as plt
from torchvision import transforms


def make_transformer_image(size_h, size_w, image_mean, image_std):
    transformer = transforms.Compose(
        [
            transforms.Resize((size_h, size_w)),  # scaling images to fixed size
            transforms.ToTensor(),  # converting to tensors
            transforms.Normalize(
                image_mean, image_std
            ),  # normalize image data per-channel
        ]
    )

    return transformer


def load_data(data_path, transformer):
    # load dataset using torchvision.datasets.ImageFolder
    train_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, "train_11k"), transform=transformer
    )
    val_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, "val"), transform=transformer
    )
    test_dataset = torchvision.datasets.ImageFolder(
        os.path.join(data_path, "test_labeled"), transform=transformer
    )
    n_train, n_val, n_test = len(train_dataset), len(val_dataset), len(test_dataset)

    return train_dataset, val_dataset, test_dataset, n_train, n_val, n_test


def batch_generator(dataset, batch_size, num_workers, shuffle=True):
    batch_gen = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers
    )

    return batch_gen


def plot_from_batch_generator(dataset, batch_size, num_workers, image_std, image_mean):
    batch_gen = batch_generator(dataset, batch_size, num_workers)
    data_batch, label_batch = next(iter(batch_gen))
    grid_size = (3, 3)
    f, ax_arr = plt.subplots(*grid_size)
    f.set_size_inches(15, 10)
    class_names = batch_gen.dataset.classes
    for i in range(grid_size[0] * grid_size[1]):
        # read images from batch to numpy.ndarray and change axes order [H, W, C] -> [H, W, C]
        batch_image_ndarray = np.transpose(data_batch[i].numpy(), [1, 2, 0])

        # inverse normalization for image data values back to [0,1] and clipping the values for correct pyplot.imshow()
        src = np.clip(image_std * batch_image_ndarray + image_mean, 0, 1)

        # display batch samples with labels
        sample_title = "Label = %d (%s)" % (label_batch[i], class_names[label_batch[i]])
        ax_arr[i // grid_size[0], i % grid_size[0]].imshow(src)
        ax_arr[i // grid_size[0], i % grid_size[0]].set_title(sample_title)

    pass
