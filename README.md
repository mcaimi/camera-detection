## Object Detection with OpenCV & YOLO

A toy work-in-progress example on how to run object detection on images/streaming media with OpenCV, using a finetuned YOLO model for detection.

The project is composed of mainly two distinct pieces:

- A real-time object detection example that analyzes a video stream from a webcam and uses OpenCV and YOLO to perform object detection
- A couple of Jupyter Notebooks that experiment with YOLO finetuning and OpenCV processing

At this point, I am using a public Drone Image dataset found on Kaggle to finetune the YOLO model.

### Kaggle credentials

Create a `.env` file under the `jupyter/` folder with these contents:

```bash
# API keys to interact with kaggle repositories
KAGGLE_USERNAME="kaggle_username"
KAGGLE_KEY="kaggle_apikey"
KAGGLEHUB_CACHE="local_cache_path"
```

### Elyra Pipelines

Pipeline for automatic/scheduled finetuning are being developed under Jupyter and Elyra on Openshift AI.

Currently, there is the need for manual secret creation for pipelines to work:

- a secret called 'yolo-kaggle' containing username/key pairs to access kaggle APIs
- a secret called 'huggingface-secret' containing the HuggingFace API Token

For example:

```bash
$ oc create secret generic yolo-kaggle --from-literal=KAGGLE_USERNAME=username --from-literal=KAGGLE_KEY=apikey --from-literal=DATASET_NAME=kaggle_dataset_name
$ oc create secret generic huggingface-secret --from-literal=HF_TOKEN=hf_api_token --from-literal=HF_HOME=hf_home_path
```

Also, the 'pipeline-pvc' persistent volume needs to be created manually before any run.

Elyra pipeline steps also need to be configured in Jupyter before running them. In particular, step 3 and 4 of the pipeline need adjustment in environment variables to suit to the environment.

### TODO

- Clean code, as it is mostly a toy implementation
- Write a detection example app
- Automate manual tasks especially pipeline requirements
