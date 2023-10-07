import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
from matplotlib import pyplot as plt
import tensorflow as tf
import numpy as np

label_names = ["Cardboard", "Glass bottle", "Can", "Crushed can", "Plastic bottle"]

model = tf.keras.models.load_model("./models/recycle_model.h5")

holdout_dir = "holdout_images"

predictions = []

for img_name in os.listdir(holdout_dir):
    img_path = os.path.join(holdout_dir, img_name)

    img = image.load_img(img_path, target_size=(128, 128))

    img_array = image.img_to_array(img)
    img_array = preprocess_input(img_array)

    img_batch = np.expand_dims(img_array, axis=0)

    predicted_scores = model.predict(img_batch)
    predicted_label = np.argmax(predicted_scores)

    plt.figure(figsize=(5, 5))
    plt.imshow(image.array_to_img(img_array))
    plt.title(f"Predicted Label: {label_names[predicted_label]}")
    plt.axis("off")
    plt.show()
