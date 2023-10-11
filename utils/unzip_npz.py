import numpy as np
import os
import matplotlib.pyplot as plt

data = np.load("recycle_data_shuffled.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

merged_dir = "unziped_images"
os.makedirs(merged_dir, exist_ok=True)

for idx, img in enumerate(x_train):
    output_path = os.path.join(merged_dir, f"train_image_{idx}.png")
    plt.imsave(output_path, img)

for idx, img in enumerate(x_test):
    output_path = os.path.join(merged_dir, f"test_image_{idx}.png")
    plt.imsave(output_path, img)
