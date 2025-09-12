# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Dataset,
    Artifact,
    Model
)


@component(base_image='python:3.11')
def unzip_dataset(
    data_dir: str,
    dataset: Input[Dataset],
    dataset_properties: Output[Artifact]
):
    # import zipfile lib
    import json
    from pathlib import PosixPath
    from zipfile import ZipFile

    with ZipFile(dataset.path, 'r') as compressed_dataset:
        compressed_dataset.extractall(data_dir)

    # save dataset properties
    properties = {
        "dataset_filename": dataset.path,
        "number_of_elements": len(list(PosixPath(data_dir).rglob("*")))
    }

    dataset_properties.path += ".json"
    with open(dataset_properties.path, "w") as artifact_dump:
        json.dump(properties, artifact_dump)


@component(base_image='python:3.11')
def unzip_model(
    data_dir: str,
    model: Input[Model],
    model_properties: Output[Artifact]
):
    # import zipfile lib
    import json
    import os
    from pathlib import PosixPath
    from zipfile import ZipFile

    with ZipFile(model.path, 'r') as compressed_model:
        compressed_model.extractall(data_dir)

    # save dataset properties
    properties = {
        "model_filename": model.path,
        "model_architecture": "YOLOv11",
        "framework": "torch",
    }

    model_properties.path += ".json"
    with open(model_properties.path, "w") as artifact_dump:
        json.dump(properties, artifact_dump)