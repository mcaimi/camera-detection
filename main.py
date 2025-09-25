#!/usr/bin/env python

# import libraries
try:
    import streamlit as st
    from PIL import Image
    import numpy as np
    import requests
    import json
    import os
    import matplotlib.pyplot as plt
    from matplotlib.figure import Figure
    from dotenv import load_dotenv
except Exception as e:
    print(f"{e}")

# TENSOR SHAPES
INPUT_SHAPE: list = [1,3,640,640]
OUTPUT_SHAPE: list = [1,47,8400]

# confidence threshhold
CONF_THRESHOLD: float = 0.60

# load dotenv
load_dotenv("dot_env")
INFER_ENDPOINT = os.getenv("INFER_ENDPOINT")
MODEL_NAME = os.getenv("MODEL_NAME")

# load labels
import yaml
LABEL_FILE: str = "./jupyter/labels.yaml"
with open(LABEL_FILE, "r") as f:
    labels = yaml.safe_load(f).get("names")

# Helper: convert Pillow image → (3, 640, 640) NumPy array (float32)
def preprocess_image(pil_img: Image.Image, tg_shape: list = (640,640)) -> np.ndarray:
    """
    Resize to 640×640, ensure RGB, convert to CHW layout,
    and normalise to [0, 1] (or whatever your model expects).
    """
    # Resize (keeping aspect ratio is optional – here we force 640×640)
    img_resized = pil_img.resize(tg_shape)

    # Ensure 3‑channel RGB
    img_rgb = img_resized.convert("RGB")

    # Convert to NumPy array, shape H×W×C → (640, 640, 3)
    img_np = np.asarray(img_rgb, dtype=np.float32)

    # Normalise (example: divide by 255 – adjust if your model needs different scaling)
    img_np /= 255.0

    # Transpose to C×H×W (channels‑first) → (3, 640, 640)
    img_chw = np.transpose(img_np, (2, 0, 1))

    return img_chw

# make POST request
def rest_request(infer_url, data, headers=None):
    json_data = {
        "inputs": [
            {
                "name": "images",
                "shape": INPUT_SHAPE,
                "datatype": "FP32",
                "data": data,
            }
        ]
    }
    if headers:
        response = requests.post(infer_url, json=json_data, headers=headers, verify=True)
    else:
        response = requests.post(infer_url, json=json_data, verify=True)
    return response

# datapoint visualization
def plot(img_rgb: np.ndarray, inferenceOutput, confidences, classes, labels):
    """
        plot an image with its relative object bounding boxes in overlay
    """
    # plot image
    plt.title(f"Dataset Point: {len(inferenceOutput)} objects")
    plt.imshow(img_rgb)

    # calculate bounding boxes
    axes = plt.gca()
    for i,o in enumerate(inferenceOutput):
        c = confidences[i]
        cl = classes[i]

        x1, y1, x2, y2 = o
        #label = object_classes[int(obj.cls)]
        print(f"Bounding Box: {x1}, {y1}, {x2}, {y2}")

        # add bounding box
        from matplotlib.patches import Rectangle
        axes.add_patch(Rectangle((x1,y1), x2,y2, color="white", fill=None))
        # add label
        ltext = f"{c} - {labels.get(cl)}"
        lpos = (x1, y1 - 10)
        axes.text(lpos[0], lpos[1], ltext, color="white", fontsize=12)

    # show datapoint
    return plt.gcf()

# figure to numpy
def fig2numpy(fig: Figure) -> np.ndarray:
    """
    @brief Convert a Matplotlib figure to a 4D numpy array with RGBA channels and return it
    @param fig a matplotlib figure
    @return a numpy 3D array of RGBA values
    """
    # draw the renderer
    fig.canvas.draw()
 
    # Get the RGBA buffer from the figure
    w,h = fig.canvas.get_width_height()
    buf = np.frombuffer(fig.canvas.tostring_argb(), dtype=np.uint8)
    buf.shape = (h, w, 4 )
 
    # canvas.tostring_argb give pixmap in ARGB mode. Roll the ALPHA channel to have it in RGBA mode
    buf = np.roll(buf, 3, axis=2)
    # return RGB image
    return buf[..., :3]

# pil2numpy
def pil2numpy(img: Image.Image, dtype=np.float32) -> np.ndarray:
    # convert image to rgb
    i = img.convert("RGB")
    # convert image to numpy array
    i_npy = np.asarray(i, dtype=dtype)

    # normalize
    if dtype == np.float32:
        i_npy /= 255.0

    # return 
    return i_npy

# MAIN UI BLOCK
# build streamlit UI
st.set_page_config(
    page_title="🧠 AI Image Recognition Service",
    initial_sidebar_state="collapsed",
    layout="wide",
)
st.html("assets/onnx.html")

st.write(
    """
    1️⃣ Upload a **JPG** or **PNG** image.  
    2️⃣ The app resizes it to **(3, 640, 640)** (channels-first).  
    3️⃣ The tensor is sent to an OpenVINO inference endpoint of your choice.
    """
)

uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])

# Upload FIle
if uploaded_file is not None:
    # prepare page
    original, resized, info = st.columns([2,2,1], vertical_alignment="top")

    # Show the original image
    image = Image.open(uploaded_file)
    original.image(image, caption="Original image", width="content")

    # Preprocess
    tensor = preprocess_image(image)  # shape (3,640,640)
    info.success(f"Pre-processed input tensor shape: {tensor.shape}")
    img_info = {
        "source": f"{uploaded_file}",
        "preprocessed": f"{tensor.shape}"
    }
    info.json(img_info, expanded=True)

    # Optional: visualise the resized image (after preprocessing)
    resized_vis = Image.fromarray(
        (np.transpose(tensor, (1, 2, 0)) * 255).astype(np.uint8)
    )
    resized.image(resized_vis, caption="Resized (640x640) preview", width="content")

    # Endpoint configuration (let the user fill these in)
    st.subheader("Openshift AI Model Server Endpoint Configuration")
    endpoint_url = st.text_input(
        "Inference URL",
        value=f"{INFER_ENDPOINT}/v2/models/{MODEL_NAME}/infer"
    )
    auth_token = st.text_input("Bearer token (optional)", type="password")

    # Send request
    if st.button("Run inference"):
        headers = {"Content-Type": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"

        with st.spinner("Performing Inference. Please wait..."):
            try:
                response = rest_request(endpoint_url, headers=headers, data=tensor.tolist())
                response.raise_for_status()
                result = response.json()
                st.success("✅ Inference succeeded!")
            except requests.exceptions.RequestException as e:
                st.error(f"❌ Request failed: {e}")

        json_prediction, detection = st.columns([2,2], vertical_alignment="top")
        json_prediction.json(result, expanded=False)  # pretty‑print the server response

        # examine prediction result
        if len(result.get("outputs")) > 0:
            output_data = result.get("outputs")[0]
            data = output_data.get("data")
            json_prediction.success(f"Got {len(data)} bytes in the prediction, type {type(data)}")
            # reshape output
            pred_classes = np.reshape(data, OUTPUT_SHAPE)
            json_prediction.success(f"Reshaped prediction to {pred_classes.shape}")
        else:
            json_prediction.warning("No Prediction")

        # display results
        predictions = pred_classes[0]
        boxes_data = predictions[0:4, :]
        scores_data = predictions[4:, :]

        boxes, confidences, class_ids = [], [], []

        # dimensions
        W, H, C = pil2numpy(image).shape
        iC, iW, iH = tensor.shape

        for i in range(predictions.shape[1]):
            class_scores = 1 / (1 + np.exp(-scores_data[:, i]))
            max_score = np.max(class_scores)
            class_id = np.argmax(class_scores)
            cx, cy, w_box, h_box = boxes_data[:, i]

            if max_score > CONF_THRESHOLD:
                left = int((cx - w_box / 2) * iW / W)
                top = int((cy - h_box / 2) * iH / H)
                w_px = int(w_box * iW / W)
                h_px = int(h_box * iH / H)

                boxes.append([left, top, w_px, h_px])
                confidences.append(float(max_score))
                class_ids.append(class_id)

        # image
        detection_image = plot(pil2numpy(image), boxes, confidences, class_ids, labels)
        detection.image(fig2numpy(detection_image), caption="Detected Objects", width="stretch")