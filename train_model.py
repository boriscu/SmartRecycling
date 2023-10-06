import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Load the data
data = np.load("recycle_data_shuffled/recycle_data_shuffled.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

#  Normalize the Images:
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten the Labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Combine cans with crushed cans
y_train = np.where(y_train == 3, 2, y_train)
y_test = np.where(y_test == 3, 2, y_test)

# Label names
label_names = ["Cardboard", "Glass bottle", "Can", "Plastic bottle"]

# Build the model
base_model_resnet = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)

# Freeze the base model layers
base_model_resnet.trainable = False

# Create a new model with custom classification layers
model_resnet = models.Sequential(
    [
        base_model_resnet,
        layers.Flatten(),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.5),
        layers.Dense(
            4, activation="softmax"
        ),  # There are 5 unique labels in the dataset
    ]
)

model_resnet.compile(
    optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

history = model_resnet.fit(
    x_train, y_train, epochs=60, validation_data=(x_test, y_test)
)

loss_resnet, accuracy_resnet = model_resnet.evaluate(x_test, y_test)
print(f"Test Loss: {loss_resnet:.4f}")
print(f"Test Accuracy: {accuracy_resnet:.4f}")

model_resnet.save("recycle_model.h5")

# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.ylabel("Accuracy")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.ylabel("Loss")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper left")

plt.tight_layout()
plt.show()

# Randomly select an image from the test set
random_idx = np.random.randint(0, len(x_test))
image = x_test[random_idx]
true_label = y_test[random_idx]

# Predict the label for the selected image
# Expand dimensions for the model's expected input shape
image_for_prediction = np.expand_dims(image, axis=0)
predicted_scores = model_resnet.predict(image_for_prediction)
predicted_label = np.argmax(predicted_scores)

# Plot the image with true and predicted labels
plt.imshow(image)
plt.title(
    f"True Label: {label_names[true_label]}\nPredicted Label: {label_names[predicted_label]}"
)
plt.axis("off")
plt.show()
