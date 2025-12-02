import streamlit as st
from ultralytics import YOLO
import os
import cv2
import numpy as np
from PIL import Image
import io
from agent import agent_graph
from langchain.schema import HumanMessage

st.set_page_config(page_title="Eye Conjunctiva Segmentation", page_icon="👁️")

st.title("👁️ Eye Conjunctiva Segmentation")
st.markdown("Advanced segmentation analysis for eye conjunctiva regions")

# Path to the trained model
MODEL_PATH = 'models/eye_conjuntiva_detection_model.pt'

# Class names and dark colors (from dataset/data.yaml)
CLASS_NAMES = ['forniceal', 'forniceal_palpebral', 'palpebral']
# Dark colors in BGR format for OpenCV
CLASS_COLORS = [
    (0, 0, 139),    # Dark red/blue for forniceal
    (0, 100, 0),    # Dark green for forniceal_palpebral
    (139, 0, 139)   # Dark magenta for palpebral
]

def create_segmentation_visualization(image, results):
    """Create segmentation-only visualization with dark colors and external labels with arrows"""
    annotated_image = image.copy()
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()  # shape: (num_instances, H, W)
        classes = results.boxes.cls.cpu().numpy().astype(int)  # class indices for each mask
        for mask, cls_idx in zip(masks, classes):
            if cls_idx < len(CLASS_COLORS):
                color = CLASS_COLORS[cls_idx]
                mask = (mask > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(annotated_image, dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = mask_resized * color[c]
                # Blend the colored mask with the original image (dark overlay, higher alpha)
                annotated_image = cv2.addWeighted(annotated_image, 1.0, colored_mask, 0.6, 0)
                # Find centroid
                moments = cv2.moments(mask_resized)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    label = CLASS_NAMES[cls_idx]
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
                    font_scale = 4
                    font_thickness = 4
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    highlight_color = (255, 255, 200)  # Light yellow
                    padding_x, padding_y = 12, 16
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

# Load the trained model
@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

st.sidebar.header("Settings")
st.sidebar.markdown(f"**Model**: Eye Conjunctiva Detection")
st.sidebar.markdown(f"**Categories**: {', '.join(CLASS_NAMES)}")

# Color legend in sidebar
st.sidebar.subheader("Color Legend")
for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
    st.sidebar.markdown(f"**{name}**: Dark color (BGR: {color})")

# Image upload
uploaded_file = st.file_uploader("Choose an eye image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display original image
    image = Image.open(uploaded_file)
    image = image.convert('RGB')
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
    
    if st.button("🔍 Analyze Eye Conjunctiva", type="primary"):
        with st.spinner("Processing eye conjunctiva segmentation..."):
            try:
                # Convert to numpy array
                img_array = np.array(image)
                
                # Run inference
                results = model(img_array, verbose=False)
                
                # Create segmentation-only visualization
                annotated_image = create_segmentation_visualization(img_array, results[0])
                
                # Convert BGR to RGB for display
                annotated_image_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                with col2:
                    st.subheader("Segmentation Result")
                    st.image(annotated_image_rgb, use_container_width=True)
                
                # Display results information
                st.divider()
                st.subheader("📊 Analysis Results")
                
                if hasattr(results[0], 'masks') and results[0].masks is not None:
                    num_detections = len(results[0].masks)
                    st.success(f"✅ Successfully segmented {num_detections} conjunctiva region(s)")
                    
                    # Get detailed information
                    masks = results[0].masks.data.cpu().numpy()
                    classes = results[0].boxes.cls.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    # Display each detection
                    cols = st.columns(min(num_detections, 3))
                    for i, (mask, cls_idx, conf) in enumerate(zip(masks, classes, confidences)):
                        with cols[i % 3]:
                            class_name = CLASS_NAMES[cls_idx]
                            color = CLASS_COLORS[cls_idx]
                            
                            # Calculate area
                            mask_binary = (mask > 0.5).astype(np.uint8)
                            area_pixels = np.sum(mask_binary)
                            
                            st.markdown(f"""
                            <div style='padding: 15px; border-radius: 10px; border: 2px solid rgb({color[2]}, {color[1]}, {color[0]}); margin-bottom: 10px;'>
                                <h4 style='color: rgb({color[2]}, {color[1]}, {color[0]}); margin: 0;'>{class_name.upper()}</h4>
                                <p style='margin: 5px 0;'><strong>Confidence:</strong> {conf:.2%}</p>
                                <p style='margin: 5px 0;'><strong>Area:</strong> {area_pixels:,} pixels</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    st.divider()
                    st.markdown("**Segmentation masks applied with dark colors:**")
                    for name, color in zip(CLASS_NAMES, CLASS_COLORS):
                        st.markdown(f"- **{name}**: Dark color (BGR: {color})")
                    
                    # Download button
                    st.divider()
                    buffered = io.BytesIO()
                    Image.fromarray(annotated_image_rgb).save(buffered, format="PNG")
                    st.download_button(
                        label="📥 Download Segmented Image",
                        data=buffered.getvalue(),
                        file_name=f"conjunctiva_segmented_{uploaded_file.name}",
                        mime="image/png",
                        type="primary"
                    )
                    
                    # AI-Powered Medical Recommendations
                    st.divider()
                    st.subheader("🤖 AI Medical Analysis & Recommendations")
                    
                    with st.spinner("Generating detailed medical analysis and recommendations..."):
                        try:
                            # Prepare detailed segmentation data for AI analysis
                            analysis_data = []
                            for i, (mask, cls_idx, conf) in enumerate(zip(masks, classes, confidences)):
                                class_name = CLASS_NAMES[cls_idx]
                                mask_binary = (mask > 0.5).astype(np.uint8)
                                area_pixels = np.sum(mask_binary)
                                total_pixels = mask.shape[0] * mask.shape[1]
                                coverage = (area_pixels / total_pixels) * 100
                                
                                analysis_data.append({
                                    'region': class_name,
                                    'confidence': conf,
                                    'coverage': coverage,
                                    'area': area_pixels
                                })
                            
                            # Create comprehensive query for AI
                            query = f"""
Based on eye conjunctiva segmentation analysis, the following regions were detected:

"""
                            for data in analysis_data:
                                query += f"- **{data['region'].upper()}**: Confidence {data['confidence']:.1%}, Coverage {data['coverage']:.2f}% ({data['area']:,} pixels)\n"
                            
                            query += """

Please provide:
1. **Medical Interpretation**: What do these segmented conjunctiva regions indicate about eye health?
2. **Clinical Significance**: Explain the importance of each detected region (forniceal, forniceal_palpebral, palpebral)
3. **Health Recommendations**: Provide specific recommendations based on the segmentation results
4. **Warning Signs**: What symptoms or conditions should be monitored?
5. **When to Seek Medical Care**: Under what circumstances should professional medical attention be sought?
6. **Preventive Measures**: What can be done to maintain healthy conjunctiva?

Please provide clear, actionable, and medically sound advice.
"""
                            
                            # Get AI response
                            response = agent_graph.invoke({
                                "messages": [HumanMessage(content=query)],
                                "user_id": "eye_conjunctiva_user"
                            })
                            
                            ai_response = response["messages"][-1].content
                            
                            # Display AI analysis in an attractive format
                            st.markdown(f"""
                            <div style='background-color: #f0f8ff; padding: 20px; border-radius: 10px; border-left: 5px solid #1e90ff;'>
                                {ai_response}
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Additional quick recommendations
                            st.divider()
                            st.subheader("⚡ Quick Recommendations")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.info("""
                                **👁️ Regular Monitoring**
                                - Check eyes daily for changes
                                - Note any redness or irritation
                                - Track symptoms in a diary
                                """)
                            
                            with col2:
                                st.success("""
                                **💧 Eye Care Tips**
                                - Use artificial tears if dry
                                - Avoid rubbing eyes
                                - Maintain proper hygiene
                                """)
                            
                            with col3:
                                st.warning("""
                                **🏥 When to Consult**
                                - Persistent redness
                                - Vision changes
                                - Discharge or pain
                                """)
                            
                        except Exception as e:
                            st.error(f"Error generating AI recommendations: {str(e)}")
                            st.info("Unable to generate AI recommendations at this time. Please consult a healthcare professional for detailed analysis.")
                
                else:
                    st.warning("⚠️ No conjunctiva regions detected in the image")
                
            except Exception as e:
                st.error(f"Error during analysis: {str(e)}")
                st.write("Please make sure you uploaded a valid eye image.")

# Information section
st.divider()
st.subheader("ℹ️ About Eye Conjunctiva Segmentation")
st.markdown("""
This tool performs automated segmentation of eye conjunctiva regions using a trained YOLOv11 model.

**Detected Regions:**
- **Forniceal**: The conjunctival fornix region
- **Forniceal Palpebral**: The transitional area between forniceal and palpebral regions  
- **Palpebral**: The conjunctiva covering the inner eyelid surface

**Features:**
- High-precision segmentation with dark color overlays
- External labels with arrows for clear identification
- Detailed metrics (confidence, area coverage)
- Downloadable annotated images

**Usage Tips:**
- Upload clear, well-lit eye images
- Ensure the conjunctiva is visible in the image
- For best results, use images similar to training data
""")
