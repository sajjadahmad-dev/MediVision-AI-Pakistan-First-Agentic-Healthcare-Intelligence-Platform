import streamlit as st
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from ultralytics import YOLO
from agent import agent_graph
from langchain.schema import HumanMessage
from PIL import Image
import io
import cv2
import numpy as np

# Eye conjunctiva detection configuration
EYE_CONJUNCTIVA_CLASS_NAMES = ['forniceal', 'forniceal_palpebral', 'palpebral']
EYE_CONJUNCTIVA_CLASS_COLORS = [
    (0, 0, 139),    # Dark red/blue for forniceal
    (0, 100, 0),    # Dark green for forniceal_palpebral
    (139, 0, 139)   # Dark magenta for palpebral
]

def create_eye_conjunctiva_visualization(image, results):
    """Create segmentation visualization for eye conjunctiva with dark colors and external labels with arrows"""
    annotated_image = image.copy()
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()  # shape: (num_instances, H, W)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # class indices for each mask
        
        for mask, cls_idx in zip(masks, classes):
            if cls_idx < len(EYE_CONJUNCTIVA_CLASS_COLORS):
                color = EYE_CONJUNCTIVA_CLASS_COLORS[cls_idx]
                mask = (mask > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]), 
                                         interpolation=cv2.INTER_NEAREST)
                
                # Create colored mask
                colored_mask = np.zeros_like(annotated_image, dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = mask_resized * color[c]
                
                # Blend the colored mask with the original image
                annotated_image = cv2.addWeighted(annotated_image, 1.0, colored_mask, 0.6, 0)
                
                # Find centroid for label placement
                moments = cv2.moments(mask_resized)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    label = EYE_CONJUNCTIVA_CLASS_NAMES[cls_idx]
                    
                    # Place label outside the mask (above and to the right)
                    offset_x, offset_y = 60, -40
                    label_x = min(cx + offset_x, annotated_image.shape[1] - 10)
                    label_y = max(cy + offset_y, 20)
                    
                    # Draw arrow from label to centroid
                    cv2.arrowedLine(
                        annotated_image,
                        (label_x, label_y),
                        (cx, cy),
                        color,
                        2,
                        tipLength=0.2
                    )
                    
                    # Draw label background (light highlight)
                    font_scale = 0.6
                    font_thickness = 2
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    highlight_color = (255, 255, 200)  # Light yellow
                    padding_x, padding_y = 8, 8
                    cv2.rectangle(
                        annotated_image,
                        (label_x - padding_x, label_y - text_size[1] - padding_y),
                        (label_x + text_size[0] + padding_x, label_y + padding_y),
                        highlight_color, -1
                    )
                    
                    # Draw label text (dark color)
                    cv2.putText(
                        annotated_image,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),  # Black text
                        font_thickness,
                        cv2.LINE_AA
                    )
    return annotated_image

st.set_page_config(page_title="MediVision AI - Image Analysis", page_icon="📸")

st.title("📸 Image Analysis")
st.markdown("Upload medical images for AI-powered object detection and description across multiple specialized models.")

# User ID
user_id = st.sidebar.text_input("User ID", value="demo_user", key="user_id")

# AI Agent Chat Section in Sidebar
st.sidebar.header("🤖 AI Medical Assistant")
st.sidebar.markdown("Ask about symptoms, health advice, find pharmacies, doctors, etc.")

# Initialize session state for chat
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Chat interface in sidebar
user_input = st.sidebar.text_input("Your message:", key="user_input", placeholder="Ask me anything...")

if st.sidebar.button("Send", key="send_button") and user_input:
    # Add user message to history
    st.session_state.chat_history.append(HumanMessage(content=user_input))

    # Invoke agent
    with st.spinner("Thinking..."):
        response = agent_graph.invoke({
            "messages": st.session_state.chat_history,
            "user_id": user_id
        })

    # Add agent response to history
    agent_response = response["messages"][-1].content
    st.session_state.chat_history.append(response["messages"][-1])

    # Display response in sidebar
    st.sidebar.write("**MediVision AI:**")
    st.sidebar.write(agent_response)

# Display Chat History in sidebar
if st.session_state.chat_history:
    st.sidebar.header("Chat History")
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.sidebar.write(f"**You:** {msg.content}")
        else:
            st.sidebar.write(f"**MediVision AI:** {msg.content}")

# Model configurations
models = {
    "Bone Detection": {
        "path": "models/bone_detection_model.pt",
        "description": "Detect bone fractures and abnormalities in X-ray images",
        "icon": "🦴"
    },
    "Brain Tumor Segmentation": {
        "path": "models/brain_tumor_segmentation_model.pt",
        "description": "Segment and identify brain tumors in MRI scans",
        "icon": "🧠"
    },
    "Eye Conjunctiva Detection": {
        "path": "models/eye_conjuntiva_detection_model.pt",
        "description": "Analyze conjunctiva regions in eye images",
        "icon": "👁️"
    },
    "Liver Disease Detection": {
        "path": "models/liver_disease_detection_model.pt",
        "description": "Detect liver diseases and abnormalities",
        "icon": "🫀"
    },
    "Skin Disease Detection": {
        "path": "models/skin_disease_detection_model.pt",
        "description": "Identify skin conditions and diseases",
        "icon": "🩹"
    },
    "Teeth Detection": {
        "path": "models/teeth_detection_model.pt",
        "description": "Detect dental issues and tooth conditions",
        "icon": "🦷"
    }
}

# Create sections for each model
for model_name, model_info in models.items():
    with st.container():
        st.header(f"{model_info['icon']} {model_name}")
        st.markdown(f"*{model_info['description']}*")

        # Create a box-like container using columns and styling
        with st.expander(f"Analyze with {model_name}", expanded=False):
            col1, col2 = st.columns([2, 1])

            with col1:
                uploaded_file = st.file_uploader(
                    f"Choose a medical image for {model_name}",
                    type=["jpg", "jpeg", "png"],
                    key=f"uploader_{model_name}"
                )

            with col2:
                st.markdown("### Quick Info")
                st.info(f"Model: {model_name}")
                st.write(f"**Purpose:** {model_info['description']}")

            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                # Convert to RGB to ensure 3 channels
                image = image.convert('RGB')
                st.image(image, caption="Uploaded Image", use_container_width=True)

                if st.button(f"🔍 Analyze with {model_name}", key=f"analyze_{model_name}"):
                    with st.spinner("Detecting objects..."):
                            try:
                                # Load YOLO model
                                model = YOLO(model_info["path"])

                                # Convert PIL to numpy array
                                img_array = np.array(image)

                                # Handle different image shapes
                                if len(img_array.shape) == 2:  # Grayscale
                                    img_array = np.stack([img_array] * 3, axis=-1)
                                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:  # RGBA
                                    img_array = img_array[:, :, :3]  # Remove alpha channel
                                elif len(img_array.shape) == 3 and img_array.shape[2] == 1:  # Single channel
                                    img_array = np.concatenate([img_array] * 3, axis=-1)

                                # Ensure uint8 dtype
                                img_array = img_array.astype(np.uint8)

                                # Run inference
                                results = model(img_array)

                                # Get detections (handles both boxes and masks)
                                detections = []
                                segmentation_info = []

                                for result in results:
                                    # Check for segmentation masks first
                                    if hasattr(result, 'masks') and result.masks is not None:
                                        masks = result.masks.data.cpu().numpy()
                                        classes = result.boxes.cls.cpu().numpy().astype(int)
                                        confidences = result.boxes.conf.cpu().numpy()

                                        for i, (mask, cls_idx, conf) in enumerate(zip(masks, classes, confidences)):
                                            class_name = model.names[cls_idx]
                                            detections.append(f"{class_name} (confidence: {conf:.2f})")

                                            # Calculate mask area
                                            mask_binary = (mask > 0.5).astype(np.uint8)
                                            area_pixels = np.sum(mask_binary)
                                            total_pixels = mask_binary.shape[0] * mask_binary.shape[1]
                                            area_percentage = (area_pixels / total_pixels) * 100

                                            segmentation_info.append({
                                                'class': class_name,
                                                'confidence': conf,
                                                'area_percentage': area_percentage,
                                                'area_pixels': area_pixels
                                            })

                                    # Fallback to regular bounding boxes if no masks
                                    elif result.boxes is not None:
                                        boxes = result.boxes
                                        for box in boxes:
                                            cls = int(box.cls[0])
                                            conf = float(box.conf[0])
                                            class_name = model.names[cls]
                                            detections.append(f"{class_name} (confidence: {conf:.2f})")

                                # Display detection results
                                if model_name == "Eye Conjunctiva Detection" and segmentation_info:
                                    st.subheader("🔍 Segmentation Results:")
                                    st.success(f"✅ Detected {len(segmentation_info)} conjunctiva region(s)")

                                    # Create columns for better layout
                                    for i, seg_info in enumerate(segmentation_info, 1):
                                        with st.expander(f"Region {i}: {seg_info['class'].upper()}", expanded=True):
                                            col1, col2 = st.columns(2)
                                            with col1:
                                                st.metric("Confidence", f"{seg_info['confidence']:.2%}")
                                                st.metric("Coverage", f"{seg_info['area_percentage']:.2f}%")
                                            with col2:
                                                st.metric("Area (pixels)", f"{seg_info['area_pixels']:,}")

                                                # Color indicator
                                                color_idx = EYE_CONJUNCTIVA_CLASS_NAMES.index(seg_info['class'])
                                                color = EYE_CONJUNCTIVA_CLASS_COLORS[color_idx]
                                                st.markdown(f"**Color**: RGB({color[2]}, {color[1]}, {color[0]})")
                                else:
                                    st.subheader("Detected Objects:")
                                    if detections:
                                        for detection in detections:
                                            st.write(f"- {detection}")
                                    else:
                                        st.write("No objects detected.")

                                # Display annotated image
                                try:
                                    if model_name == "Eye Conjunctiva Detection" and segmentation_info:
                                        st.subheader("📊 Segmentation Visualization:")

                                        # Use custom eye conjunctiva visualization
                                        annotated_img = create_eye_conjunctiva_visualization(img_array, results[0])
                                        # Convert BGR to RGB for display
                                        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                                        st.image(annotated_img, caption="Eye Conjunctiva Segmentation with Labels", use_container_width=True)

                                        # Show detailed color legend
                                        st.subheader("🎨 Color Legend:")
                                        legend_cols = st.columns(len(EYE_CONJUNCTIVA_CLASS_NAMES))
                                        for idx, (name, color) in enumerate(zip(EYE_CONJUNCTIVA_CLASS_NAMES, EYE_CONJUNCTIVA_CLASS_COLORS)):
                                            with legend_cols[idx]:
                                                st.markdown(f"""
                                                <div style='padding: 10px; border-radius: 5px; background-color: rgb({color[2]}, {color[1]}, {color[0]}); color: white; text-align: center;'>
                                                    <strong>{name.upper()}</strong>
                                                </div>
                                                """, unsafe_allow_html=True)

                                        # Add download button for segmented image
                                        st.divider()
                                        buffered = io.BytesIO()
                                        Image.fromarray(annotated_img).save(buffered, format="PNG")
                                        st.download_button(
                                            label="📥 Download Segmented Image",
                                            data=buffered.getvalue(),
                                            file_name=f"segmented_{uploaded_file.name}",
                                            mime="image/png"
                                        )
                                    else:
                                        # Use default YOLO visualization for other models
                                        annotated_img = results[0].plot()
                                        # Convert BGR to RGB for display
                                        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)
                                        st.image(annotated_img, caption="Annotated Image", use_container_width=True)
                                except Exception as e:
                                    st.error(f"Error displaying annotated image: {str(e)}")

                                # Send to LLM for description
                                if detections or segmentation_info:
                                    st.divider()

                                    if model_name == "Eye Conjunctiva Detection" and segmentation_info:
                                        # Create detailed query for eye conjunctiva
                                        regions = [f"{info['class']} (coverage: {info['area_percentage']:.1f}%)" for info in segmentation_info]
                                        query = f"Based on the eye conjunctiva segmentation analysis, the following regions were detected: {', '.join(regions)}. Explain the medical significance of these conjunctiva regions and what this segmentation reveals about eye health."
                                    else:
                                        detection_text = ", ".join(detections)
                                        query = f"Describe the medical significance of these detections in the image: {detection_text}"

                                    with st.spinner("Generating AI medical analysis..."):
                                        try:
                                            response = agent_graph.invoke({
                                                "messages": [HumanMessage(content=query)],
                                                "user_id": user_id
                                            })

                                            st.subheader("🤖 AI Medical Analysis:")
                                            st.markdown(response["messages"][-1].content)
                                        except Exception as e:
                                            st.error(f"Error generating description: {str(e)}")
                                else:
                                    st.info("No detections to describe.")

                            except Exception as e:
                                st.error(f"Error during analysis: {str(e)}")
                                st.write(f"Image shape: {np.array(image).shape}")
                                st.write(f"Image mode: {image.mode}")
