import numpy as np


def combined_data_generator(x, y, batch_size, datagen):
    data_gen = datagen.flow(
        x, y, batch_size=batch_size // 2
    )  # Half batch size for augmented data
    while True:
        # Get a batch of augmented data
        x_augmented, y_augmented = next(data_gen)

        # Get a batch of original data
        idx = np.random.choice(len(x), batch_size // 2, replace=False)
        x_original = x[idx]
        y_original = y[idx]

        # Combine original and augmented data
        x_combined = np.concatenate([x_original, x_augmented])
        y_combined = np.concatenate([y_original, y_augmented])

        # Shuffle the combined data
        indices = np.arange(batch_size)
        np.random.shuffle(indices)
        x_combined = x_combined[indices]
        y_combined = y_combined[indices]

        yield x_combined, y_combined
