#!/usr/bin/env python3
"""People detection on the live camera feed using YOLOv4-tiny."""

import argparse
import time
from collections import deque

import cv2
import numpy as np

DEVICE = "/dev/video0"
WIDTH = 1280
HEIGHT = 720
TARGET_FPS = 30

MAX_FAILURES = 30
RECONNECT_DELAY = 2.0
INTERVAL_WINDOW = 300

CFG = "models/yolov4-tiny.cfg"
WEIGHTS = "models/yolov4-tiny.weights"
NAMES = "models/coco.names"

INPUT_SIZE = 416
CONF_THRESHOLD = 0.4
NMS_IOU = 0.4
PERSON_CLASS_ID = 0

GREEN = (0, 255, 0)


class Detection:
    """One detected object, in frame pixel coordinates."""

    def __init__(self, x, y, width, height, confidence, label):
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.label = label

    def top_left(self):
        return (self.x, self.y)

    def bottom_right(self):
        return (self.x + self.width, self.y + self.height)


def load_class_names(path):
    """Read coco.names into a list of 80 strings."""
    text_file = open(path)
    contents = text_file.read()
    text_file.close()

    contents = contents.strip()
    return contents.split("\n")


def load_network(use_cuda):
    """Load YOLOv4-tiny and choose where it runs."""
    net = cv2.dnn.readNetFromDarknet(CFG, WEIGHTS)

    if use_cuda:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA)
        print("requested CUDA backend")
    else:
        net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    return net


def open_camera(device, width, height, fps):
    cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
    if not cap.isOpened():
        return None

    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def describe_camera(cap):
    """What the camera actually gives not what we requested."""
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = int(cap.get(cv2.CAP_PROP_FPS))
    return f"{actual_width}x{actual_height} @ {actual_fps}fps"


def run_network(net, output_layers, frame):
    """Feed one frame through the network. Returns the raw output arrays."""
    blob = cv2.dnn.blobFromImage(
        frame,
        1 / 255.0,
        (INPUT_SIZE, INPUT_SIZE),
        swapRB=True,
        crop=False,
    )
    net.setInput(blob)
    return net.forward(output_layers)


def extract_boxes(outputs, frame_width, frame_height,
                       conf_threshold, person_only):
    """Turn raw network output into three parallel lists.

    Each row of the output is: centre x, centre y, width, height,
    objectness, then one score per class.
    """
    boxes = []
    confidences = []
    class_ids = []

    for output in outputs:
        for row in output:
            class_scores = row[5:]
            best_class_id = int(np.argmax(class_scores))
            best_score = float(class_scores[best_class_id])

            if best_score < conf_threshold:
                continue

            if person_only and best_class_id != PERSON_CLASS_ID:
                continue

            # Coordinates arrive normalized 0-1 and centre-based.
            centre_x = row[0] * frame_width
            centre_y = row[1] * frame_height
            box_width = row[2] * frame_width
            box_height = row[3] * frame_height

            left = int(centre_x - box_width / 2)
            top = int(centre_y - box_height / 2)

            boxes.append([left, top, int(box_width), int(box_height)])
            confidences.append(best_score)
            class_ids.append(best_class_id)

    return boxes, confidences, class_ids


def merge_overlapping(boxes, confidences, conf_threshold):
    """Non-maximum suppression. Returns the indices worth keeping."""
    if not boxes:
        return []

    kept = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, NMS_IOU)
    kept_array = np.array(kept)
    flat = kept_array.flatten()
    return flat.tolist()


def detect(net, output_layers, class_names, frame,
           conf_threshold, person_only):
    """Run detection on one frame. Returns a list of Detection objects."""
    frame_height = frame.shape[0]
    frame_width = frame.shape[1]

    outputs = run_network(net, output_layers, frame)

    boxes, confidences, class_ids = extract_boxes(
        outputs, frame_width, frame_height, conf_threshold, person_only)

    kept_indices = merge_overlapping(boxes, confidences, conf_threshold)

    detections = []
    for index in kept_indices:
        left, top, box_width, box_height = boxes[index]
        label = class_names[class_ids[index]]
        detections.append(
            Detection(left, top, box_width, box_height,
                      confidences[index], label))

    return detections


def draw_detections(frame, detections):
    for detection in detections:
        cv2.rectangle(frame, detection.top_left(),
                      detection.bottom_right(), GREEN, 2)

        text = f"{detection.label} {detection.confidence:.2f}"
        text_y = max(detection.y - 8, 14)
        cv2.putText(frame, text, (detection.x, text_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, GREEN, 2)


def average_ms(nanosecond_samples):
    """Average a deque of nanosecond durations, in milliseconds."""
    if not nanosecond_samples:
        return 0.0

    total = sum(nanosecond_samples)
    average_nanoseconds = total / len(nanosecond_samples)
    return average_nanoseconds / 1_000_000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="use when display-less")
    parser.add_argument("--threshold", type=float, default=CONF_THRESHOLD,
                        help="minimum confidence to report")
    parser.add_argument("--detect-every", type=int, default=1,
                        help="run detection every Nth frame to speed up")
    parser.add_argument("--person-only", action="store_true",
                        help="ignore everything except people")
    parser.add_argument("--cuda", action="store_true",
                        help="try the CUDA backend instead of CPU")
    return parser.parse_args()


def main():
    args = parse_args()

    net = load_network(args.cuda)
    class_names = load_class_names(NAMES)
    output_layers = net.getUnconnectedOutLayersNames()
    print(f"model loaded, {len(class_names)} classes")

    cap = None
    frame_count = 0
    consecutive_failures = 0
    frame_intervals = deque(maxlen=INTERVAL_WINDOW)
    detect_times = deque(maxlen=INTERVAL_WINDOW)
    last_frame_time = None
    detections = []

    try:
        while True:
            if cap is None:
                cap = open_camera(DEVICE, WIDTH, HEIGHT, TARGET_FPS)

                if cap is None:
                    print(f"waiting for {DEVICE}...")
                    time.sleep(RECONNECT_DELAY)
                    continue

                print(f"actual: {describe_camera(cap)}")
                consecutive_failures = 0
                last_frame_time = None

            ret, frame = cap.read()

            if not ret:
                consecutive_failures += 1
                if consecutive_failures >= MAX_FAILURES:
                    print(f"{consecutive_failures} failed reads, reopening")
                    cap.release()
                    cap = None
                continue

            consecutive_failures = 0
            frame_count += 1

            now = time.monotonic_ns()
            if last_frame_time is not None:
                frame_intervals.append(now - last_frame_time)
            last_frame_time = now

            if frame_count % args.detect_every == 0:
                started = time.monotonic_ns()
                detections = detect(net, output_layers, class_names, frame,
                                    args.threshold, args.person_only)
                detect_times.append(time.monotonic_ns() - started)

            draw_detections(frame, detections)

            if args.headless:
                if frame_count % 60 == 0:
                    cv2.imwrite(f"tmp/det_{frame_count:05d}.jpg", frame)
            else:
                cv2.imshow("detect", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frame_count % 30 == 0:
                interval = average_ms(frame_intervals)
                detect_time = average_ms(detect_times)
                print(f"frames: {frame_count}  "
                      f"interval: {interval:.1f}ms  "
                      f"detect: {detect_time:.1f}ms  "
                      f"objects: {len(detections)}")

    except KeyboardInterrupt:
        print("\nstopped")

    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        if cap is not None:
            cap.release()


if __name__ == "__main__":
    main()
