from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from PIL import Image
import io
import os
import cv2
import numpy as np
from ultralytics import YOLO
import openai
import base64
import json
from datetime import datetime

app = FastAPI(
    title="MediVision AI - Model API",
    description="API for medical image analysis using various YOLO models.",
    version="1.0.0",
)

# Define model paths
MODELS = {
    "bone_detection_model": "models/bone_detection_model.pt",
    "brain_tumor_segmentation_model": "models/brain_tumor_segmentation_model.pt",
    "eye_conjunctiva_detection_model": "models/eye_conjunctiva_detection_model.pt",
    "liver_disease_detection_model": "models/liver_disease_detection_model.pt",
    "skin_disease_detection_model": "models/skin_disease_detection_model.pt",
    "teeth_detection_model": "models/teeth_detection_model.pt"
}

# Eye conjunctiva specific configuration (copied from pages/eye_conjunctiva.py)
EYE_CONJUNCTIVA_CLASS_NAMES = ['forniceal', 'forniceal_palpebral', 'palpebral']
EYE_CONJUNCTIVA_CLASS_COLORS = [
    (0, 0, 139),    # Dark red/blue for forniceal
    (0, 100, 0),    # Dark green for forniceal_palpebral
    (139, 0, 139)   # Dark magenta for palpebral
]

# Function to create custom eye conjunctiva visualization (copied from pages/eye_conjunctiva.py)
def create_eye_conjunctiva_visualization(image_np, results):
    annotated_image = image_np.copy()
    if hasattr(results, 'masks') and results.masks is not None:
        masks = results.masks.data.cpu().numpy()
        classes = results.boxes.cls.cpu().numpy().astype(int)
        for mask, cls_idx in zip(masks, classes):
            if cls_idx < len(EYE_CONJUNCTIVA_CLASS_COLORS):
                color = EYE_CONJUNCTIVA_CLASS_COLORS[cls_idx]
                mask = (mask > 0.5).astype(np.uint8)
                mask_resized = cv2.resize(mask, (annotated_image.shape[1], annotated_image.shape[0]), interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(annotated_image, dtype=np.uint8)
                for c in range(3):
                    colored_mask[:, :, c] = mask_resized * color[c]
                annotated_image = cv2.addWeighted(annotated_image, 1.0, colored_mask, 0.6, 0)
                moments = cv2.moments(mask_resized)
                if moments["m00"] != 0:
                    cx = int(moments["m10"] / moments["m00"])
                    cy = int(moments["m01"] / moments["m00"])
                    label = EYE_CONJUNCTIVA_CLASS_NAMES[cls_idx]
                    offset_x, offset_y = 60, -40
                    label_x = min(cx + offset_x, annotated_image.shape[1] - 10)
                    label_y = max(cy + offset_y, 20)
                    cv2.arrowedLine(
                        annotated_image,
                        (label_x, label_y),
                        (cx, cy),
                        color,
                        2,
                        tipLength=0.2
                    )
                    font_scale = 0.6 # Adjusted for API output, not Streamlit
                    font_thickness = 2
                    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0]
                    highlight_color = (255, 255, 200)
                    padding_x, padding_y = 8, 8
                    cv2.rectangle(
                        annotated_image,
                        (label_x - padding_x, label_y - text_size[1] - padding_y),
                        (label_x + text_size[0] + padding_x, label_y + padding_y),
                        highlight_color, -1
                    )
                    cv2.putText(
                        annotated_image,
                        label,
                        (label_x, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale,
                        (0, 0, 0),
                        font_thickness,
                        cv2.LINE_AA
                    )
    return annotated_image

# Load models into memory
# This is a simple approach; for production, consider lazy loading or a dedicated model service
LOADED_MODELS = {}

def get_model(model_name: str):
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail="Model not found")
    if model_name not in LOADED_MODELS:
        try:
            LOADED_MODELS[model_name] = YOLO(MODELS[model_name])
            print(f"Loaded model: {model_name}")
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error loading model {model_name}: {str(e)}")
    return LOADED_MODELS[model_name]

@app.get("/")
async def read_root():
    return {"message": "Welcome to MediVision AI Model API. Use /docs for API documentation."}

@app.post("/analyze/{model_name}")
async def analyze_image(model_name: str, file: UploadFile = File(...)):
    """
    Analyzes an uploaded medical image using the specified YOLO model.
    Returns the annotated image and AI-generated medical analysis.
    """
    if model_name not in MODELS:
        raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found. Available models: {', '.join(MODELS.keys())}")

    try:
        # Read image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        img_array = np.array(image)

        # Load model
        model = get_model(model_name)

        # Perform inference
        results = model(img_array, verbose=False) # verbose=False to suppress YOLO output

        # Process detections and generate AI analysis query
        detections = []
        segmentation_info = []
        ai_query = ""

        for result in results:
            if hasattr(result, 'masks') and result.masks is not None:
                masks = result.masks.data.cpu().numpy()
                classes = result.boxes.cls.cpu().numpy().astype(int)
                confidences = result.boxes.conf.cpu().numpy()
                
                for i, (mask, cls_idx, conf) in enumerate(zip(masks, classes, confidences)):
                    class_name = model.names[cls_idx]
                    detections.append(f"{class_name} (confidence: {conf:.2f})")
                    
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
            elif result.boxes is not None:
                boxes = result.boxes
                for box in boxes:
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    class_name = model.names[cls]
                    detections.append(f"{class_name} (confidence: {conf:.2f})")

        if model_name == "eye_conjunctiva_detection_model" and segmentation_info:
            regions = [f"{info['class']} (coverage: {info['area_percentage']:.1f}%)" for info in segmentation_info]
            ai_query = f"""
Based on eye conjunctiva segmentation analysis, the following regions were detected:

"""
            for data in segmentation_info:
                ai_query += f"- **{data['class'].upper()}**: Confidence {data['confidence']:.1%}, Coverage {data['area_percentage']:.2f}% ({data['area_pixels']:,} pixels)\n"

            ai_query += """

Please provide:
1. **Medical Interpretation**: What do these segmented conjunctiva regions indicate about eye health?
2. **Clinical Significance**: Explain the importance of each detected region (forniceal, forniceal_palpebral, palpebral)
3. **Health Recommendations**: Provide specific recommendations based on the segmentation results
4. **Warning Signs**: What symptoms or conditions should be monitored?
5. **When to Seek Medical Care**: Under what circumstances should professional medical attention be sought?
6. **Preventive Measures**: What can be done to maintain healthy conjunctiva?

Please provide clear, actionable, and medically sound advice.
"""
        elif detections:
            detection_text = ", ".join(detections)
            ai_query = f"""
Describe the medical significance of these detections in the image: {detection_text}.

Please provide:
1. **Medical Interpretation**: What do these detections indicate about the patient's health?
2. **Clinical Significance**: Explain the importance of these findings
3. **Recommendations**: Provide specific recommendations based on the detection results
4. **Suggestions**: What are the suggested next steps for the patient?
5. **Warning Signs**: What symptoms or conditions should be monitored?
6. **When to Seek Medical Care**: Under what circumstances should professional medical attention be sought?
7. **Preventive Measures**: What can be done to prevent or manage related conditions?

Please provide clear, actionable, and medically sound advice.
"""
        else:
            ai_query = """
No specific medical objects or conditions detected in the image. However, please provide general recommendations and suggestions for medical image analysis and health monitoring.

Please provide:
1. **General Recommendations**: What are the best practices for medical imaging?
2. **Health Suggestions**: General advice for maintaining health when no specific conditions are detected
3. **Monitoring**: What should patients monitor in their health?
4. **Preventive Measures**: General preventive healthcare measures
5. **When to Seek Care**: When should someone consult a healthcare professional even without specific detections?

Please provide clear, actionable, and medically sound advice.
"""

        # Generate AI medical analysis using OpenAI
        ai_analysis_content = "No AI analysis generated."
        if ai_query:
            try:
                import os
                api_key = os.getenv("OPENAI_API_KEY")
                if not api_key:
                    ai_analysis_content = "OpenAI API key not configured. Please set OPENAI_API_KEY environment variable."
                else:
                    client = openai.OpenAI(api_key=api_key)
                    response = client.chat.completions.create(
                        model="gpt-4",
                        messages=[
                            {"role": "system", "content": "You are a medical AI assistant providing professional medical analysis and recommendations based on image analysis results."},
                            {"role": "user", "content": ai_query}
                        ],
                        max_tokens=1000,
                        temperature=0.3
                    )
                    ai_analysis_content = response.choices[0].message.content
            except Exception as e:
                print(f"Error generating AI analysis: {e}")
                ai_analysis_content = f"Error generating AI analysis: {str(e)}"

        # Generate annotated image
        annotated_img_np = None
        if model_name == "eye_conjunctiva_detection_model" and segmentation_info:
            annotated_img_np = create_eye_conjunctiva_visualization(img_array, results[0])
            annotated_img_np = cv2.cvtColor(annotated_img_np, cv2.COLOR_BGR2RGB)
        else:
            annotated_img_np = results[0].plot()
            annotated_img_np = cv2.cvtColor(annotated_img_np, cv2.COLOR_BGR2RGB)

        # Save annotated image to a buffer
        img_buffer = io.BytesIO()
        Image.fromarray(annotated_img_np).save(img_buffer, format="PNG")
        img_buffer.seek(0)

        # Return both image and JSON analysis
        return JSONResponse(content={
            "model_name": model_name,
            "detections": detections,
            "segmentation_info": segmentation_info,
            "ai_analysis": ai_analysis_content,
            "annotated_image_base64": base64.b64encode(img_buffer.getvalue()).decode("utf-8")
        })

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
