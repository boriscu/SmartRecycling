import time

import cv2
import imageio
import numpy as np
import tensorflow as tf
from keras.preprocessing.image import load_img, img_to_array
from keras.applications.resnet50 import preprocess_input
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


def detect_objects_path(image_path):
    image_np = np.array(load_img(image_path))
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detector(input_tensor)
    return detections["detection_boxes"], detections["detection_scores"]


def detect_objects_image(image_np):
    input_tensor = tf.convert_to_tensor(image_np)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = detector(input_tensor)
    return detections["detection_boxes"], detections["detection_scores"]


def classify_image(image_array):
    preprocessed_image = preprocess_input(image_array)

    # show_preprocessed_image(preprocessed_image)

    classification = resnet_model.predict(np.expand_dims(preprocessed_image, axis=0))
    class_idx = np.argmax(classification)
    class_prob = classification[0][class_idx]

    return class_idx, class_prob


def process_and_visualize(image_path):
    boxes, scores = detect_objects_path(image_path)
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


def process_visualise_image(image_np):
    boxes, scores = detect_objects_image(image_np)

    for i, box in enumerate(boxes[0].numpy()):
        score = scores[0].numpy()[i]
        if score < 0.45:
            continue

        ymin, xmin, ymax, xmax = box
        cropped_region = image_np[
            int(ymin * image_np.shape[0]) : int(ymax * image_np.shape[0]),
            int(xmin * image_np.shape[1]) : int(xmax * image_np.shape[1]),
        ]

        height, width, _ = cropped_region.shape

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

        xmin, ymin, xmax, ymax = (
            xmin * image_np.shape[1],
            ymin * image_np.shape[0],
            xmax * image_np.shape[1],
            ymax * image_np.shape[0],
        )

        cv2.rectangle(
            image_np, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 0, 255), 2
        )

        class_name = label_names[cls]
        cv2.putText(
            image_np,
            f"{class_name} {prob * 100:.2f}%",
            (int(xmin), int(ymin) - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

    return image_np


# process_and_visualize("../holdout_images/plasticna_flasa_naopacke.jpg")
def detect_objects_in_live_video():
    print("Running live video inference...")

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    output_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    output_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 10
    video_writer = cv2.VideoWriter(
        "live_annotated_video.mp4",
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (output_width, output_height),
    )

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Error: Could not read frame.")
            break

        annotated_frame = process_visualise_image(frame)

        video_writer.write(annotated_frame)

        cv2.imshow("Live Video", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    video_writer.release()
    cv2.destroyAllWindows()


def detect_objects_in_video(input_video):
    print(f"Running inference for {input_video}.mp4... ", end="")

    video_reader = imageio.get_reader(f"{input_video}.mp4")
    video_writer = imageio.get_writer(f"{input_video}_annotated.mp4", fps=10)

    # loop through and process each frame
    t0 = time.time()
    n_frames = 0
    for frame in video_reader:
        n_frames += 1
        new_frame = process_visualise_image(frame)

        # instead of plotting image, we write the frame to video
        video_writer.append_data(new_frame)

    fps = n_frames / (time.time() - t0)
    print("Frames processed: %s, Speed: %s fps" % (n_frames, fps))

    # clean up
    video_writer.close()


if __name__ == "__main__":
    # detect_objects_in_live_video()
    detect_objects_in_video("../videos/test")
