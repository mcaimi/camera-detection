# Import objects from KubeFlow DSL Library
from kfp.dsl import (
    component,
    Input,
    Output,
    Model,
)


@component(base_image="quay.io/marcocaimi/ultralytics-onnx:latest")
def convert_model(
    data_dir: str,
    finetuned_model: Input[Model],
    onnx_model: Output[Model],
):
    from zipfile import ZipFile
    from pathlib import PosixPath
    import os
    
    # Unzip source model
    workdir: str = "/".join((data_dir, "onnx"))
    os.makedirs(workdir, exist_ok=True)
    with ZipFile(finetuned_model.path, 'r') as compressed_model:
        compressed_model.extractall(workdir)

    # convert & save model to onnx
    from ultralytics import YOLO
    weights_file: str = max(PosixPath(workdir).rglob("last.pt"), key=os.path.getmtime)
    torch_model = YOLO(weights_file)
    torch_model.export(format="onnx")
    
    # save model to s3
    onnx_model._set_path(onnx_model.path + "-onnx.zip")
    last_file: str = max(PosixPath(workdir).rglob("*.onnx"), key=os.path.getmtime)

    # zip & store
    import zipfile
    with zipfile.ZipFile(onnx_model.path, "w", zipfile.ZIP_DEFLATED) as zip_file:
        zip_file.write(last_file)