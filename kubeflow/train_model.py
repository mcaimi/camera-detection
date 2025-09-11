# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Model
)


@component(base_image="ultralytics/ultralytics:latest")
def train_model(
    original_model: Input[Model],
    dataset: Input[Dataset],
    hyperparameters: dict,
    data_mount_path: str,
    finetuned_model: Output[Model],
):
    import ultralytics
    import yaml
    import json
    import os
    import torch.cuda as tc
    import pprint

    # Training Parameters
    JOB = hyperparameters.get("job")
    RUN_NAME = hyperparameters.get("run_name")
    TRAINING_CONFIG = hyperparameters.get("training_job_descriptor")
    CHECKPOINT = hyperparameters.get("checkpoint")
    EPOCHS = hyperparameters.get("epochs")
    LR = hyperparameters.get("learning_rate")
    IMG_SIZE = hyperparameters.get("img_size")
    BATCH = hyperparameters.get("batch")
    OPTIMIZER = hyperparameters.get("optimizer")
    AUGMENT = hyperparameters.get("augment")

    # detect device
    device = "cpu"
    if tc.is_available():
        device = "cuda"

    cp: str = f"{data_mount_path}/{CHECKPOINT}"
    tc: str = f"{data_mount_path}/{TRAINING_CONFIG}"
    print(f"DEVICE:\n Training model {cp} on {device}")

    # load model
    yolo_model = ultralytics.YOLO(cp)
    yolo_model.to(device)

    # fix training descriptor
    with open(tc, "r") as training_descriptor_r:
        training_parms = yaml.safe_load(training_descriptor_r)
    # update base path
    training_parms["path"] = os.path.dirname(tc)
    # dump parms
    pprint.pprint(training_parms)
    # write descriptor back
    with open(tc, "w") as training_descriptor_w:
        yaml.dump(training_parms, training_descriptor_w)

    # fix run_dir path
    with open("/.config/Ultralytics/settings.json", "r") as yolo_settings:
        ys = json.load(yolo_settings)
    ys["runs_dir"] = f"{data_mount_path}/runs/{JOB}/{RUN_NAME}"
    ys["datasets_dir"] = f"{data_mount_path}/datasets/{JOB}/{RUN_NAME}"
    ys["weights_dir"] = f"{data_mount_path}/weights/{JOB}/{RUN_NAME}"
    pprint.pprint(ys)
    with open("/.config/Ultralytics/settings.json", "w") as yolo_settings:
        json.dump(ys, yolo_settings)

    # start training!
    yolo_model.train(data=tc,
                     epochs=EPOCHS, lr0=LR, imgsz=IMG_SIZE, batch=BATCH,
                     resume=False, optimizer=OPTIMIZER, augment=AUGMENT,
                     project=data_mount_path)

    # validate
    training_metrics = yolo_model.val()

    # convert to ONNX
    yolo_model.export(format="onnx")