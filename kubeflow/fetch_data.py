# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Output,
    Dataset,
    Model,
)


@component(base_image='python:3.11',
           packages_to_install=["kagglehub",
                                "python-dotenv",
                                ])
def fetch_data(
    dataset_name: str,
    version: str,
    dataset: Output[Dataset]
):
    # import libs
    import os
    try:
        import kagglehub
    except Exception as e:
        print(f"Caught exception {e}")

    # Check environment
    KG_USER = os.getenv('KAGGLE_USERNAME')
    KG_PASS = os.getenv('KAGGLE_KEY')
    KG_PATH = os.getenv('KAGGLEHUB_CACHE')

    print(f"Connecting to Kaggle as {KG_USER} with API Key {KG_PASS}, save artifacts to {KG_PATH}")

    # download dataset from kaggle now
    DATASET_NAME: str = dataset_name
    TRAINING_DATASET_PATH: str = KG_PATH

    # get the dataset
    print(f"Dataset '{DATASET_NAME}' will be downloaded to {TRAINING_DATASET_PATH}")
    dspath: str = kagglehub.dataset_download(DATASET_NAME)
    print(f"Dataset available @{dspath}")

    # zip the dataset
    import zipfile
    from pathlib import Path

    # save output dataset to S3
    dataset.path += ".zip"
    srcdir = Path(dspath)

    with zipfile.ZipFile(dataset.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in srcdir.rglob("*"):
            zip_file.write(entry, entry.relative_to(srcdir))


@component(base_image='python:3.11',
           packages_to_install=["huggingface_hub"])
def fetch_model(
    model_name: str,
    hyperparameters: dict,
    original_model: Output[Model],
):
    try:
        import os
        import zipfile
        from pathlib import Path
        import huggingface_hub as hf
    except Exception as e:
        raise e

    HF_TOKEN: str = os.getenv("HF_TOKEN")

    # Download model checkpoint from HuggingFace repositories
    yolo_path: str = "/".join(("/tmp/", model_name))
    os.makedirs(yolo_path, exist_ok=True)

    print(f"Downloading model checkpoint: {model_name}")
    model_path = hf.snapshot_download(repo_id=model_name,
                                    allow_patterns=hyperparameters.get("checkpoint"),
                                    revision="main",
                                    token=HF_TOKEN,
                                    local_dir=yolo_path)

    # save output dataset to S3
    original_model._set_path(original_model.path + ".zip")
    srcdir = Path(yolo_path)

    with zipfile.ZipFile(original_model.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        for entry in srcdir.rglob("*"):
            zip_file.write(entry, entry.relative_to(srcdir))