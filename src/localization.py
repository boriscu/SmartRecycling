import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import tensorflow_hub as hub

label_names = ["Cardboard", "Glass", "Metal", "Plastic"]

model_url = "https://tfhub.dev/tensorflow/ssd_mobilenet_v2/2"
detector = hub.load(model_url)

resnet_model = tf.keras.models.load_model("../models/recycle_model.h5")


def show_preprocessed_image(preprocessed_image):
    adjusted_image = (
        (preprocessed_image - preprocessed_image.min())
        / (preprocessed_image.max() - preprocessed_image.min())
        * 255
    )
    adjusted_image = adjusted_image.astype("uint8")

    plt.imshow(adjusted_image)
    plt.title("Preprocessed Input to Classification Model")
    plt.axis("off")
    plt.show()


def detect_objects(image_path):
    image_np = np.array(load_img(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detector(input_tensor)
    return detections["detection_boxes"], detections["detection_scores"]


def classify_image(image_array):
    preprocessed_image = preprocess_input(image_array)

    show_preprocessed_image(preprocessed_image)

    classification = resnet_model.predict(np.expand_dims(preprocessed_image, axis=0))
    class_idx = np.argmax(classification)
    class_prob = classification[0][class_idx]

    return class_idx, class_prob


def process_and_visualize(image_path):
    boxes, scores = detect_objects(image_path)
    max_score_idx = np.argmax(
        scores[0].numpy()
    )  # Index of the highest confidence score
    box = boxes[0][
        max_score_idx
    ].numpy()  # Select the box with the highest confidence score
    score = scores[0].numpy()[max_score_idx]

    if score < 0.5:
        print("Warning! Localization confidence is below 0.5")

    image = load_img(image_path)
    image_array = img_to_array(image)
    ymin, xmin, ymax, xmax = box

    cropped_region = image_array[
        int(ymin * image_array.shape[0]) : int(ymax * image_array.shape[0]),
        int(xmin * image_array.shape[1]) : int(xmax * image_array.shape[1]),
    ]

    height, width, _ = cropped_region.shape
    if height != width:
        diff = abs(height - width)
        if height < width:
            padding_top = diff // 2
            padding_bottom = diff - padding_top
            padding = ((padding_top, padding_bottom), (0, 0), (0, 0))
        else:
            padding_left = diff // 2
            padding_right = diff - padding_left
            padding = ((0, 0), (padding_left, padding_right), (0, 0))

        square_region = np.pad(cropped_region, padding, mode="constant")
        cropped_region_resized = tf.image.resize(square_region, (128, 128))
        cls, prob = classify_image(cropped_region_resized.numpy())
    else:
        cls, prob = classify_image(cropped_region)

    # Visualization
    plt.imshow(image)
    ax = plt.gca()
    rect = patches.Rectangle(
        (xmin * image_array.shape[1], ymin * image_array.shape[0]),
        (xmax - xmin) * image_array.shape[1],
        (ymax - ymin) * image_array.shape[0],
        linewidth=1,
        edgecolor="r",
        facecolor="none",
    )
    ax.add_patch(rect)
    plt.text(
        xmin * image_array.shape[1],
        ymin * image_array.shape[0],
        f"{label_names[cls]} {(prob*100):.2f}%",
        bbox=dict(facecolor="white", alpha=0.5),
    )
    plt.show()


process_and_visualize("../holdout_images/plasticna_flasa.jpg")
