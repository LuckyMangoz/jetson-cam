#!/usr/bin/env python3
"""Camera capture loop."""

import argparse
import time
from collections import deque

import cv2

DEVICE = "/dev/video0"
WIDTH = 1280
HEIGHT = 720
TARGET_FPS = 30


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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true",
                        help="save periodic frames instead of opening a window")

    # parser.add_argument("--format", default="MJPG", choices=["MJPG", "YUYV"])
    # parser.add_argument("--width", type=int, default=1280)
    # parser.add_argument("--height", type=int, default=720)

    args = parser.parse_args()


    cap = open_camera(DEVICE, WIDTH, HEIGHT, TARGET_FPS)
    if cap is None:
        print(f"Could not open {DEVICE}")
        return

    print(f"actual: {cap.get(cv2.CAP_PROP_FRAME_WIDTH):.0f}"
          f"x{cap.get(cv2.CAP_PROP_FRAME_HEIGHT):.0f}"
          f" @ {cap.get(cv2.CAP_PROP_FPS):.0f}fps")

    frames = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("read failed")
                break

            frames += 1

            if args.headless:
                if frames % 60 == 0:
                    cv2.imwrite(f"/tmp/frame_{frames:05d}.jpg", frame)
            else:
                cv2.imshow("capture", frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if frames % 30 == 0:
                print(f"frames: {frames}")

    except KeyboardInterrupt:
        print("\nstopped")
    finally:
        if not args.headless:
            cv2.destroyAllWindows()
        cap.release()


if __name__ == "__main__":
    main()
