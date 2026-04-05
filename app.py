import streamlit as st
import torch
import cv2
import numpy as np
import math
from PIL import Image
import segmentation_models_pytorch as smp

# --- 1. App Configuration ---
st.set_page_config(page_title="AFCA AI Dashboard", layout="wide")
st.title("Auto-Foliculometry Clinical Assistant (AFCA)")
st.write("Upload a transvaginal ultrasound scan for instant, AI-powered follicle biometry.")

# --- 2. Load the AI Brain (Cached so it only loads once) ---
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
    # Make sure your .pth file is in the same folder as this script!
    model.load_state_dict(torch.load("afca_unet_model.pth", map_location=device))
    model.eval()
    return model, device

model, device = load_model()

# --- 3. Clinical Variables ---
PIXEL_TO_MM = 0.25 

# --- 4. User Upload Interface ---
uploaded_file = st.file_uploader("Upload Ultrasound Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Read the image the user uploaded
    image = Image.open(uploaded_file).convert('RGB')
    raw_image = np.array(image)
    
    # Show loading spinner while AI thinks
    with st.spinner("Analyzing scan and computing biometry..."):
        
        # Format image for the U-Net
        resized_image = cv2.resize(raw_image, (256, 256))
        image_tensor = resized_image / 255.0
        image_tensor = np.transpose(image_tensor, (2, 0, 1))
        image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(device)

        # Run AI Prediction
        with torch.no_grad():
            prediction = model(image_tensor)
            prediction = torch.sigmoid(prediction)
            predicted_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

        # OpenCV Biometry and Drawing
        output_image = resized_image.copy()
        contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        mature_count = 0
        total_count = 0

        for contour in contours:
            area_px = cv2.contourArea(contour)
            if area_px < 50: # Ignore noise
                continue
                
            total_count += 1
            radius_px = math.sqrt(area_px / math.pi)
            diameter_mm = (radius_px * 2) * PIXEL_TO_MM
            
            # Clinical Color Coding
            if diameter_mm >= 18.0:
                color = (0, 255, 0) # Green for mature
                mature_count += 1
            elif diameter_mm >= 10.0:
                color = (255, 255, 0) # Yellow for growing
            else:
                color = (255, 0, 0) # Red for baseline
                
            cv2.drawContours(output_image, [contour], -1, color, 2)
            
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                cv2.putText(output_image, f"{diameter_mm:.1f}mm", (cX - 15, cY), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

    # --- 5. Display the Final Dashboard ---
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Original Scan")
        st.image(raw_image, use_container_width=True)
        
    with col2:
        st.subheader("AFCA Automated Biometry")
        st.image(output_image, use_container_width=True)
        
    st.success(f"**Clinical Summary:** Detected {total_count} total follicles. {mature_count} follicles are >= 18mm (Ready for IVF Trigger).")