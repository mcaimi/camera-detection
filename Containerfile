FROM docker.io/ultralytics/ultralytics:latest

RUN conda install -y onnx onnxslim onnxruntime
