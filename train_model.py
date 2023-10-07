from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
        layers.Dense(
            1024,
            activation="relu",
            kernel_initializer="he_normal",
            kernel_regularizer=tf.keras.regularizers.l2(0.005),
        ),
        layers.Dense(512, activation="relu", kernel_initializer="he_normal"),
        layers.Dropout(0.3),
        layers.Dense(256, activation="relu", kernel_initializer="he_normal"),
        layers.Dense(5, activation="softmax"),
    ]
)

# Define a learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,  # Starting learning rate
    decay_steps=1000,  # Decay the learning rate after every 1000 steps
    decay_rate=0.9,  # Decay rate (lr=initial*decay every 1000 steps)
)

# Instantiate the optimizer with the learning rate schedule
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model_resnet.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
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


# Callbacks
early_stop = EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True)
checkpoint = ModelCheckpoint(
    "best_weights.h5", save_best_only=True, monitor="val_accuracy", mode="max"
)

# Fit the model
batch_size = 50
initial_epochs = 40
combined_gen = combined_data_generator(x_train, y_train, batch_size, datagen)
history = model_resnet.fit(
    combined_gen,
    steps_per_epoch=len(x_train) / batch_size,
    epochs=initial_epochs,
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpoint],
)

# Unfreeze the top 10 layers of the base model
for layer in base_model_resnet.layers[-10:]:
    layer.trainable = True

# Recompile the model after unfreezing
model_resnet.compile(
    optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"]
)

# Continue training (fine-tuning)
fine_tune_epochs = 20
total_epochs = initial_epochs + fine_tune_epochs

history_fine = model_resnet.fit(
    combined_gen,
    steps_per_epoch=len(x_train) / batch_size,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],  # Continue from where we left off
    validation_data=(x_test, y_test),
    callbacks=[early_stop, checkpoint],
)

model_resnet.save("recycle_model_fine_tuned.h5")

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

# Plot training & validation accuracy values for fine training
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history_fine["accuracy"])
plt.plot(history.history_fine["val_accuracy"])
plt.title("Model Fine Accuracy")
plt.ylabel("Fine Accuracy")
plt.xlabel("Fine Epoch")
plt.legend(["Fine Train", "Fine Test"], loc="upper left")

# Plot training & validation loss values for fine training
plt.subplot(1, 2, 2)
plt.plot(history.history_fine["loss"])
plt.plot(history.history_fine["val_loss"])
plt.title("Model Fine Loss")
plt.ylabel("Fine Loss")
plt.xlabel("Fine Epoch")
plt.legend(["Fine Train", "Fine Test"], loc="upper left")

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

# Saving the model to google drive
# from google.colab import drive
# drive.mount('/content/drive')
# model_save_path = "/content/drive/MyDrive/Modeli/recycle_model.h5"
# model_resnet.save(model_save_path)
