import torch
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from io import BytesIO

# Function to convert Matplotlib figure to image for download
def plt_to_image(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    return buf.read()

# Load pre-trained Mask R-CNN model
model = models.detection.maskrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Image transformations
transform = transforms.Compose([
    transforms.ToTensor()
])

# Streamlit app setup
st.set_page_config(page_title="Mask R-CNN Image Segmentation", page_icon=":guardsman:", layout="wide")

# Header
st.title("Mask R-CNN Image Segmentation")
st.sidebar.header("Settings")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # Display image preview
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Confidence threshold slider
    confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

    # Show progress bar while processing
    with st.spinner("Processing the image..."):
        # Transform the image
        input_image = transform(image).unsqueeze(0)  # Add batch dimension

        # Perform prediction
        with torch.no_grad():
            outputs = model(input_image)

        # Extract predictions
        boxes = outputs[0]["boxes"]
        labels = outputs[0]["labels"]
        scores = outputs[0]["scores"]
        masks = outputs[0]["masks"]

        # Filter predictions based on confidence threshold
        filtered_indices = [i for i, score in enumerate(scores) if score > confidence_threshold]
        filtered_boxes = boxes[filtered_indices]
        filtered_masks = masks[filtered_indices]

        # Create a matplotlib figure for visualization
        fig, ax = plt.subplots(figsize=(12, 8))
        ax.imshow(image)

        for i, box in enumerate(filtered_boxes):
            x1, y1, x2, y2 = box
            ax.add_patch(plt.Rectangle((x1, y1), x2 - x1, y2 - y1, linewidth=2, edgecolor="yellow", facecolor="none"))
            mask = filtered_masks[i, 0].cpu().numpy()
            ax.imshow(mask, cmap="inferno", alpha=0.5)  # Smoother color palette for mask

        ax.axis("off")

        # Display the result in Streamlit
        st.pyplot(fig)

        # Allow user to download the result
        st.sidebar.download_button("Download Segmented Image", data=plt_to_image(fig), file_name="segmented_image.png", mime="image/png")
