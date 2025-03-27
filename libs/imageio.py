#!/usr/bin/env python

try:
    import cv2 as cv
    import numpy as np
    import torch
except Exception as e:
    print(f"Cannot load library: {e}")
    raise e

COLOR_WHITE = (255, 255, 255)


# draw capture info
def displayFrameInfo(frame: np.ndarray, font, x, y) -> None:
    # image statistics
    frame_h = frame.shape[0]
    frame_w = frame.shape[1]
    frame_channels = frame.shape[2]

    cv.putText(frame,
               "Capturing... ctrl+c to exit",
               (x, y + 24), font, 1, COLOR_WHITE, 2, cv.LINE_AA)
    cv.putText(frame,
               f"Frame Size: {int(frame_w)}px x {int(frame_h)}px @ {int(frame_channels)} channels",
               (x, y + 64), font, 1, COLOR_WHITE, 2, cv.LINE_AA)


# get random color in RGB format
def getRandomColor() -> list:
    # get a random 32-bit integer
    rand_int = np.random.randint(0, 0xFFFFFF)

    # return R, G, B sequence
    return (rand_int & 0xFF, (rand_int >> 8) & 0xFF, (rand_int >> 16) & 0xFF)


# extract color planes from image
def components(image: np.ndarray) -> list[np.ndarray]:
    h, w = image.shape[:2]

    # extract component planes from an image
    components: list = [image[:, :, c] for c in range(image.shape[2])]
    # null ndarray plane
    z: np.ndarray = np.zeros((h, w), dtype=image.dtype)

    # return data planes
    return [np.stack([components[0], z, z], axis=-1),
            np.stack([z, components[1], z], axis=-1),
            np.stack([z, z, components[2]], axis=-1)]


# split RGB channels
def splitRGBChannels(image: np.ndarray) -> list[np.ndarray]:
    # get color channels
    r, g, b = cv.split(image)

    # return channels
    return [r, g, b]


# split HSV channels
def splitHSVChannels(image: np.ndarray) -> list[np.ndarray]:
    # convert image from RGB to HSV
    image_hsv: np.ndarray = cv.cvtColor(image, cv.COLOR_RGB2HSV)

    # get channels
    h, s, v = cv.split(image_hsv)

    # return channels
    return [h, s, v]


# overlay analyze detected features
def detectedObjects(inferenceOutput, targetFrame):
    for o in inferenceOutput:
        object_classes = o.names
        for obj in o.boxes:
            if type(obj.xyxy) is torch.Tensor:
                # determine object coordinates
                bbox = obj.xyxy.cpu().type(torch.int32).numpy()
                x1, y1, x2, y2 = bbox[0]
                # draw object detection box
                cv.rectangle(targetFrame, (x1, y1), (x2, y2), (255, 0, 0), 3)

                # determine object classification label and confidence score
                class_label = object_classes[int(obj.cls)]
                confidence = float(obj.conf)
                # print detected object class
                cv.putText(targetFrame, f"{class_label} - {confidence:.2f}",
                            (x1, y1 - 20),
                            cv.FONT_HERSHEY_SIMPLEX,
                            1,
                            (255, 255, 255),
                            1,
                            2)

    # return frame
    return targetFrame

