import io
import cv2
import streamlit as st
from ultralytics import YOLO
import time

# Hide main menu style
menu_style_cfg = """<style>MainMenu {visibility: hidden;}</style>"""

# Main title of streamlit application
main_title_cfg = """<div><h1 style="color:#FF64DA; text-align:center; font-size:40px; 
                         font-family: 'Archivo', sans-serif; margin-top:-50px;margin-bottom:20px;">
                Real time Crack Detection App
                </h1></div>"""

# Subtitle of streamlit application
# sub_title_cfg = """<div><h4 style="color:#042AFF; text-align:center;
#                 font-family: 'Archivo', sans-serif; margin-top:-15px; margin-bottom:50px;">
#                 Experience real-time object detection on your webcam with the power of Ultralytics YOLOv8! 🚀</h4>
#                 </div>"""

# Set html page configuration
st.set_page_config(page_title="Ultralytics Streamlit App", layout="wide", initial_sidebar_state="auto")

# Append the custom HTML
st.markdown(menu_style_cfg, unsafe_allow_html=True)
st.markdown(main_title_cfg, unsafe_allow_html=True)

# Add ultralytics logo in sidebar
with st.sidebar:
    logo = "https://raw.githubusercontent.com/ultralytics/assets/main/logo/Ultralytics_Logotype_Original.svg"
    st.image(logo, width=250)

# Add elements to vertical setting menu
st.sidebar.title("User Configuration")

# Add video source selection dropdown
source = st.sidebar.selectbox(
    "Video",
    ("image", "video"),
)

if source == "video":
    vid_file = st.sidebar.file_uploader("Upload Video File", type=["mp4", "mov", "avi", "mkv"])
    if vid_file is not None:
        g = io.BytesIO(vid_file.read())  # BytesIO Object
        vid_location = "ultralytics.mp4"
        with open(vid_location, "wb") as out:  # Open temporary file as bytes
            out.write(g.read())  # Read bytes into file
        vid_file_name = "ultralytics.mp4"

if source == 'image':
    img_file = st.sidebar.file_uploader("Upload image", type=['jpg', 'jpeg', 'png'])
    if img_file is not None:
        g = io.BytesIO(img_file.read())  # BytesIO Object
        img_location = "ultralytics.jpg"
        with open(img_location, "wb") as out:  # Open temporary file as bytes
            out.write(g.read())  # Read bytes into file
        img_file_name = "ultralytics.jpg"

# Add dropdown menu for model selection
model = st.sidebar.selectbox(
    "Model",
    (
        "YOLOv8",
        "Detectron2",
         "SAM"
    ),
)
bbox = st.sidebar.selectbox(
    "Bounding Box",
    (
        "Yes",
        "No"
    ),
)
model = YOLO("best.pt")  # Load the yolov8 model

conf_thres = st.sidebar.slider("Confidence Threshold",0.0, 1.0, 0.25, 0.01)
nms_thres = st.sidebar.slider("NMS Threshold", 0.0, 1.0, 0.45, 0.01)

col1, col2 = st.columns(2)
org_frame = col1.empty()
ann_frame = col2.empty()

if st.sidebar.button("Start"):

    if source == "video":
                fps_display = st.sidebar.empty()  # Placeholder for FPS display

                videocapture = cv2.VideoCapture(vid_file_name)  # Capture the video

                if not videocapture.isOpened():
                    st.error("Could not open webcam.")
                success, frame = videocapture.read()
                stop_button = st.button("Stop")  # Button to stop the inference

                prev_time = 0
                while success:
                    success, frame = videocapture.read()
                    if not success:
                      break

                    curr_time = time.time()
                    fps = 1 / (curr_time - prev_time)
                    prev_time = curr_time
                    results = model(frame, conf=conf_thres, iou=nms_thres)
                    # Store model predictions
                    for result in results:
                        masks = result.masks  # Masks object for segmentation masks outputs

                        # Plot the results on the original image and get the image with annotations
                        annotated_image = result.plot()

                        # Display the image with annotations
                        org_frame.image(frame, channels="BGR")
                        ann_frame.image(annotated_image, channels="BGR")

                    if stop_button:
                        videocapture.release()  # Release the capture
                    # Display FPS in sidebar
                    fps_display.metric("FPS", f"{fps:.2f}")

                # Release the capture
                videocapture.release()

    if source == "image":
        results = model(img_file_name, conf=conf_thres, iou=nms_thres)

        if bbox == "Yes":

                for result in results:
                    masks = result.masks  # Masks object for segmentation masks outputs

                    # Plot the results on the original image and get the image with annotations
                    annotated_image = result.plot()

                    # Display the image with annotations
                    org_frame.image(img_file, channels="BGR")
                    ann_frame.image(annotated_image, channels="BGR")

        else:
            for result in results:
                # Extract masks and the original image
                masks = result.masks.data.cpu().numpy()  # Masks as numpy arrays
                orig_img = result.orig_img  # Original image

                # Convert the original image to RGB (if it's in BGR format)
                if len(orig_img.shape) == 3 and orig_img.shape[2] == 3:
                    image_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
                else:
                    image_rgb = orig_img

                # Create an overlay for the segmentation masks
                overlay = image_rgb.copy()

                # Apply the masks to the overlay
                for mask in masks:
                    mask_resized = cv2.resize(mask, (
                    orig_img.shape[1], orig_img.shape[0]))  # Resize mask to match image dimensions
                    overlay[mask_resized > 0.5] = (255, 0, 0)  # Apply a color to the mask area (red in this case)

                # Blend the original image with the overlay
                alpha = 0.5  # Transparency factor
                segmented_image = cv2.addWeighted(overlay, alpha, image_rgb, 1 - alpha, 0)

                # Display the image with only segmentation masks
                org_frame.image(img_file, channels="BGR")
                ann_frame.image(segmented_image, channels="BGR")

