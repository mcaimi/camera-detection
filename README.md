## Object Detection with OpenCV & YOLO

A toy work-in-progress example on how to run object detection on a live streaming feed from a webcam, using YOLO for realtime detection.

The project is composed of mainly two distinct pieces:

- A real-time object detection example that analyzes a video stream from a webcam and uses OpenCV and YOLO to perform object detection
- A couple of Jupyter Notebooks that experiment with YOLO finetuning and OpenCV processing

The Finetuning notebook uses free images of Super Mario found on Pixabay to let YOLO recognize characters from the videogame.
Labeling has been done by hand for now, using [Label Studio](https://labelstud.io/)

### TODO

- Clean code, as it is mostly a toy implementation
