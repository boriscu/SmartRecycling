import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Get the data
# !wget -nc http://web.cecs.pdx.edu/~singh/rcyc-web/recycle_data_shuffled.tar.gz
# !tar -xvzf recycle_data_shuffled.tar.gz

# Load the data
data = np.load("recycle_data_shuffled.npz")
x_train, y_train = data["x_train"], data["y_train"]
x_test, y_test = data["x_test"], data["y_test"]

#  Normalize the Images:
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0

# Flatten the Labels
y_train = y_train.flatten()
y_test = y_test.flatten()

# Label names
label_names = ["Cardboard", "Glass bottle", "Can", "Crushed can", "Plastic bottle"]

# Show a random image
random_idx = np.random.randint(0, len(x_train))
plt.imshow(x_train[random_idx])
plt.title(f"Label: {label_names[y_train[random_idx]]}")
plt.axis("off")
plt.show()

# Build the model
base_model_resnet = tf.keras.applications.ResNet50(
    weights="imagenet", include_top=False, input_shape=(128, 128, 3)
)

# Freeze the base model layers
base_model_resnet.trainable = False

model_resnet = models.Sequential(
    [
        base_model_resnet,
        layers.Flatten(),
        layers.Dense(1024, activation="relu"),
        layers.Dense(512, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu"),
        layers.Dense(5, activation="softmax"),
    ]
)

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.009,  # Starting learning rate
    decay_steps=1000,  # Decay the learning rate after every 1000 steps
    decay_rate=0.9,  # Decay rate
)

optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model_resnet.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest",
)
datagen.fit(x_train)

# Show agumented images
augmented_images, augmented_labels = next(datagen.flow(x_train, y_train, batch_size=5))
for img, label in zip(augmented_images, augmented_labels):
    plt.imshow(img)
    plt.title(label_names[label])
    plt.show()

# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "best_weights.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)

# Fit the model
batch_size = 32
history = model_resnet.fit(
    datagen.flow(x_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(x_train) / batch_size,
    epochs=100,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpoint],
)
model_resnet.save("recycle_model.h5")

# Evaluate the model
loss_resnet, accuracy_resnet = model_resnet.evaluate(x_test, y_test)
print(f"Test Loss: {loss_resnet:.4f}")
print(f"Test Accuracy: {accuracy_resnet:.4f}")


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

# Predict the label for the selected image
random_idx = np.random.randint(0, len(x_test))
image = x_test[random_idx]
true_label = y_test[random_idx]

image_for_prediction = np.expand_dims(image, axis=0)
predicted_scores = model_resnet.predict(image_for_prediction)
predicted_label = np.argmax(predicted_scores)

plt.imshow(image)
plt.title(
    f"True Label: {label_names[true_label]}\nPredicted Label: {label_names[predicted_label]}"
)
plt.axis("off")
plt.show()


# Plot confusion matrix
y_pred = model_resnet.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)

cm = confusion_matrix(y_test, y_pred_classes)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=label_names,
    yticklabels=label_names,
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()
