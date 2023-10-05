import numpy as np

# Load the data from the .npz file
data = np.load("recycle_data_shuffled/recycle_data_shuffled.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

print("Shape of x:", x_train.shape)
print("Shape of y:", y_train.shape)
print("Data type of x:", x_train.dtype)
print("Data type of y:", y_train.dtype)
print("Min and Max values of x:", x_train.min(), x_train.max())
print("Unique labels:", np.unique(y_train))

import matplotlib.pyplot as plt

random_idx = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_idx])
plt.title(f"Label: {y_train[random_idx]}")
plt.axis("off")
plt.show()
