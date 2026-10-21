
import os
import io
import cv2
import tempfile
import numpy as np
import streamlit as st
from PIL import Image
from ultralytics import SAM

# --------------------------------------------------
# APP CONFIGURATION
# --------------------------------------------------

st.set_page_config(
    page_title="SAM 2.1 Image & Video Segmentation",
    page_icon="🎯",
    layout="wide"
)

st.title("🎯 Image & Video Segmentation")
st.caption("Powered by Ultralytics SAM 2.1 — runs locally on your Mac.")

MODEL_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "sam2.1_b.pt"
)

OUTPUT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "processed_videos"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# --------------------------------------------------
# LOAD MODEL ONCE
# --------------------------------------------------

@st.cache_resource
def load_model():
    if not os.path.isfile(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found: {MODEL_PATH}"
        )

    return SAM(MODEL_PATH)


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------

def resize_frame(frame, max_width):
    height, width = frame.shape[:2]

    if width <= max_width:
        return frame

    scale = max_width / width
    new_width = max(2, int(width * scale))
    new_height = max(2, int(height * scale))

    return cv2.resize(
        frame,
        (new_width, new_height),
        interpolation=cv2.INTER_AREA
    )


def segment_frame(model, frame, x, y):
    height, width = frame.shape[:2]

    x = max(0, min(int(x), width - 1))
    y = max(0, min(int(y), height - 1))

    results = model(
        frame,
        points=[[x, y]],
        labels=[1],
        imgsz=384,
        device="cpu",
        verbose=False
    )

    return results[0].plot()


def image_to_png_bytes(image_array):
    success, encoded = cv2.imencode(".png", image_array)

    if not success:
        raise RuntimeError("Could not encode the result as PNG.")

    return encoded.tobytes()


# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.header("Settings")
st.sidebar.write("Model: SAM 2.1 Base")
st.sidebar.write("Device: CPU")
st.sidebar.write("Inference image size: 384 pixels")
st.sidebar.caption(
    "Smaller processing sizes reduce workload but may reduce detail."
)


# --------------------------------------------------
# FILE UPLOAD
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload an image or video",
    type=[
        "jpg", "jpeg", "png", "webp",
        "mp4", "mov", "avi", "mkv"
    ]
)

if uploaded_file is None:
    st.info(
        "Upload an image to segment an object, "
        "or a video to process its frames."
    )
    st.stop()


filename = uploaded_file.name
extension = os.path.splitext(filename)[1].lower()

video_extensions = {".mp4", ".mov", ".avi", ".mkv"}
is_video = extension in video_extensions

file_bytes = uploaded_file.getvalue()


# --------------------------------------------------
# IMAGE SEGMENTATION
# --------------------------------------------------

if not is_video:

    try:
        pil_image = Image.open(io.BytesIO(file_bytes)).convert("RGB")
        original = np.array(pil_image)
        height, width = original.shape[:2]

        st.subheader("Image")
        st.write(f"Image dimensions: {width} × {height}")

        preview = resize_frame(
            cv2.cvtColor(original, cv2.COLOR_RGB2BGR),
            1000
        )
        preview_rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)

        st.image(
            preview_rgb,
            caption="Original image preview",
            use_container_width=True
        )

        # Coordinates are entered relative to the original image.
        col1, col2 = st.columns(2)

        with col1:
            x = st.number_input(
                f"Object point X (0 to {width - 1})",
                min_value=0,
                max_value=width - 1,
                value=width // 2
            )

        with col2:
            y = st.number_input(
                f"Object point Y (0 to {height - 1})",
                min_value=0,
                max_value=height - 1,
                value=height // 2
            )

        st.caption(
            "Enter the coordinates of a point inside the object "
            "you want to select. Coordinates refer to the original image."
        )

        if st.button("Segment Image", type="primary"):

            with st.spinner("Loading model and segmenting image..."):
                model = load_model()

                image_bgr = cv2.cvtColor(
                    original,
                    cv2.COLOR_RGB2BGR
                )

                result = segment_frame(model, image_bgr, x, y)
                result_rgb = cv2.cvtColor(
                    result,
                    cv2.COLOR_BGR2RGB
                )

                st.subheader("Segmentation Result")
                st.image(
                    result_rgb,
                    use_container_width=True
                )

                png_bytes = image_to_png_bytes(result)

                st.download_button(
                    "Download Segmented Image",
                    data=png_bytes,
                    file_name="segmented_image.png",
                    mime="image/png"
                )

    except Exception as error:
        st.error(f"Image processing failed: {error}")


# --------------------------------------------------
# VIDEO SEGMENTATION
# --------------------------------------------------

else:

    st.subheader("Video")
    st.video(file_bytes)

    st.warning(
        "CPU processing can be slow. Start with a short clip. "
        "This version segments sampled frames independently; "
        "it does not track the same object automatically between frames."
    )

    col1, col2 = st.columns(2)

    with col1:
        max_width = st.select_slider(
            "Maximum processing width",
            options=[320, 480, 640],
            value=320
        )

    with col2:
        stride = st.select_slider(
            "Process every Nth frame",
            options=[1, 2, 3, 5, 10],
            value=3
        )

    # Save the uploaded video outside any temporary directory.
    input_path = os.path.join(OUTPUT_DIR, "uploaded_input" + extension)

    with open(input_path, "wb") as file:
        file.write(file_bytes)

    cap = cv2.VideoCapture(input_path)

    if not cap.isOpened():
        cap.release()
        st.error(
            "OpenCV could not open this video. "
            "Try a standard MP4 video encoded with H.264."
        )
        st.stop()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(fps) or fps <= 0:
        fps = 24.0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    ok, first_frame = cap.read()

    if not ok:
        cap.release()
        st.error("Could not read the first video frame.")
        st.stop()

    first_small = resize_frame(first_frame, max_width)
    frame_h, frame_w = first_small.shape[:2]

    cap.release()

    st.write(
        f"Processing resolution: {frame_w} × {frame_h}"
    )

    st.image(
        cv2.cvtColor(first_small, cv2.COLOR_BGR2RGB),
        caption="First frame — choose a point inside the object",
        use_container_width=True
    )

    col1, col2 = st.columns(2)

    with col1:
        point_x = st.number_input(
            f"Object point X (0 to {frame_w - 1})",
            min_value=0,
            max_value=frame_w - 1,
            value=frame_w // 2,
            key="video_x"
        )

    with col2:
        point_y = st.number_input(
            f"Object point Y (0 to {frame_h - 1})",
            min_value=0,
            max_value=frame_h - 1,
            value=frame_h // 2,
            key="video_y"
        )

    st.caption(
        "Choose a point inside the object in the resized preview. "
        "The same point is used for every sampled frame."
    )

    if st.button("Segment Video", type="primary"):

        output_path = os.path.join(
            OUTPUT_DIR,
            "segmented_video.avi"
        )

        cap = cv2.VideoCapture(input_path)

        if not cap.isOpened():
            st.error("Could not reopen the uploaded video.")
            st.stop()

        # MJPG in AVI is generally a practical fallback on macOS.
        output_fps = max(1.0, fps / stride)

        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*"MJPG"),
            output_fps,
            (frame_w, frame_h)
        )

        if not writer.isOpened():
            cap.release()
            st.error(
                "Could not create the output video. "
                "Your OpenCV installation may lack a compatible codec."
            )
            st.stop()

        try:
            model = load_model()

            progress = st.progress(0)
            status = st.empty()

            frame_index = 0
            processed = 0

            while True:
                ok, frame = cap.read()

                if not ok:
                    break

                if frame_index % stride == 0:
                    small_frame = resize_frame(frame, max_width)

                    # Ensure each output frame has identical dimensions.
                    if small_frame.shape[1] != frame_w or small_frame.shape[0] != frame_h:
                        small_frame = cv2.resize(
                            small_frame,
                            (frame_w, frame_h),
                            interpolation=cv2.INTER_AREA
                        )

                    result = segment_frame(
                        model,
                        small_frame,
                        point_x,
                        point_y
                    )

                    writer.write(result)
                    processed += 1

                frame_index += 1

                if total_frames > 0:
                    progress.progress(
                        min(1.0, frame_index / total_frames)
                    )

                status.write(
                    f"Frames read: {frame_index} | "
                    f"Frames segmented: {processed}"
                )

            cap.release()
            writer.release()

            if (
                not os.path.isfile(output_path)
                or os.path.getsize(output_path) == 0
                or processed == 0
            ):
                st.error(
                    "The output video was not created successfully."
                )
                st.stop()

            st.success(
                f"Video segmentation finished. "
                f"{processed} frames were segmented."
            )

            st.video(output_path)

            with open(output_path, "rb") as file:
                output_bytes = file.read()

            st.download_button(
                "Download Segmented Video (AVI)",
                data=output_bytes,
                file_name="segmented_video.avi",
                mime="video/x-msvideo"
            )

        except Exception as error:
            st.error(f"Video processing failed: {error}")

        finally:
            cap.release()
            writer.release()
