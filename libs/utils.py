#!/usr/bin/env python

try:
    from torch import float16, float32
    import cv2 as cv
    import torch.cuda as cuda
    import torch.backends.mps as apple_mps
    from ultralytics import YOLO
    from .parameters import Parameters
    from .huggingface import pullFromHuggingfaceHub
except Exception as e:
    print(f"Cannot load pytorch: {e}")
    raise e


# detect available gpu
def getAcceleratorDevice() -> (str, int):
    accelerator: str = "cpu"
    dtype = float16

    if apple_mps.is_available():
        print("Apple Metal Performance Shaders Available!")
        accelerator = "mps"
    elif cuda.is_available():
        device_name = cuda.get_device_name()
        device_capabilities = cuda.get_device_capability()
        device_available_mem, device_total_mem = [x / 1024**3 for x in cuda.mem_get_info()]
        print(f"A GPU is available! [{device_name} - {device_capabilities} - {device_available_mem}/{device_total_mem} GB VRAM]")
        accelerator = "cuda"
    else:
        print("NO GPU FOUND.")
        dtype = float32

    return accelerator, dtype


# get a capture device
def getCaptureDevice(index: int):
    return cv.VideoCapture(index)


# load model and use any detected accelerator
def loadAndPrepareModel(configParams: Parameters):
    mFiles = pullFromHuggingfaceHub(configParams)
    modelCheckpoint, _ = mFiles.popitem()
    model = YOLO(modelCheckpoint)
    return model

