## Object Detection with OpenCV & YOLO

A toy work-in-progress example on how to run object detection on images/streaming media with OpenCV, using a finetuned YOLO model for detection.

The project is composed of mainly two distinct pieces:

- A real-time object detection example that analyzes a video stream from a webcam and uses OpenCV and YOLO to perform object detection
- A couple of Jupyter Notebooks that experiment with YOLO finetuning and OpenCV processing

At this point, I am using a public Drone Image dataset found on Kaggle to finetune the YOLO model.

### Elyra Pipelines

Pipeline for automatic/scheduled finetuning are being developed under Jupyter and Elyra on Openshift AI.

Currently, there is the need for manual secret creation for pipelines to work:

- a secret called 'yolo-kaggle' containing username/key pairs to access kaggle APIs
- a secret called 'huggingface-secret' containing the HuggingFace API Token

Also, the 'pipeline-pvc' persistent volume needs to be created manually before any run.

### TODO

- Clean code, as it is mostly a toy implementation
- Write a detection example app
- Automate manual tasks especially pipeline requirements
