import streamlit as st
import sqlite3
import hashlib
import torch
import cv2
import numpy as np
import math
from PIL import Image
import segmentation_models_pytorch as smp
from fpdf import FPDF
import tempfile
import os

# --- 1. App Configuration ---
st.set_page_config(page_title="AFCA AI Dashboard", layout="wide")

# --- 2. Database Backend Setup ---
def create_usertable():
    conn = sqlite3.connect('afca_users.db')
    c = conn.cursor()
    # UNIQUE ensures no two doctors can register the same ID
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT UNIQUE, password TEXT)')
    conn.commit()

def add_userdata(username, password):
    conn = sqlite3.connect('afca_users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    try:
        c.execute('INSERT INTO userstable(username, password) VALUES (?,?)', (username, hashed_pw))
        conn.commit()
        return True
    except sqlite3.IntegrityError:
        return False # Triggers if username already exists

def login_user(username, password):
    conn = sqlite3.connect('afca_users.db')
    c = conn.cursor()
    hashed_pw = hashlib.sha256(password.encode()).hexdigest()
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?', (username, hashed_pw))
    data = c.fetchall()
    return data

create_usertable() # Initialize database on startup

# --- 3. Session State Management ---
if 'logged_in' not in st.session_state:
    st.session_state['logged_in'] = False
if 'show_signup' not in st.session_state:
    st.session_state['show_signup'] = False

# --- 4. Authentication Portal (Login / Sign Up) ---
if not st.session_state['logged_in']:
    # Hide sidebar on login screen
    st.markdown("""
        <style>
            [data-testid="collapsedControl"] {display: none;}
            section[data-testid="stSidebar"] {display: none;}
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<h1 style='text-align: center; color: #2e6c80;'>AFCA Portal</h1>", unsafe_allow_html=True)
        st.markdown("<h4 style='text-align: center; color: gray;'>Clinical Decision Support System</h4>", unsafe_allow_html=True)
        st.write("---")

        # --- SIGN UP VIEW ---
        if st.session_state['show_signup']:
            with st.form("signup_form"):
                st.subheader("Create Clinician Account")
                new_user = st.text_input("New Clinician ID / Username")
                new_pass = st.text_input("New Password", type='password')
                submit_signup = st.form_submit_button("Register Account")

            if submit_signup:
                if new_user and new_pass:
                    success = add_userdata(new_user, new_pass)
                    if success:
                        st.success("Account created successfully! You can now log in.")
                        st.session_state['show_signup'] = False
                        st.rerun()
                    else:
                        st.error("Clinician ID already exists. Please choose another.")
                else:
                    st.warning("Please fill out both fields.")
            
            if st.button("← Back to Login"):
                st.session_state['show_signup'] = False
                st.rerun()

        # --- LOGIN VIEW ---
        else:
            with st.form("login_form"):
                st.subheader("Physician Login")
                username = st.text_input("Clinician ID")
                password = st.text_input("Password", type='password')
                submit_login = st.form_submit_button("Secure Login")

            if submit_login:
                result = login_user(username, password)
                if result:
                    st.session_state['logged_in'] = True
                    st.session_state['username'] = username
                    st.rerun()
                else:
                    st.error("Invalid Clinician ID or Password. Please try again.")
            
            st.write("Don't have an account?")
            if st.button("Create New Account"):
                st.session_state['show_signup'] = True
                st.rerun()

# --- 5. Main Application Dashboard ---
else:
    # Sidebar features
    st.sidebar.success(f"Logged in as: Dr. {st.session_state['username']}")
    if st.sidebar.button("Logout"):
        st.session_state['logged_in'] = False
        st.rerun()

    # Dashboard Header
    st.title("Auto-Foliculometry Clinical Assistant (AFCA)")
    st.write("Welcome to the secure clinical dashboard. Upload a transvaginal ultrasound scan for instant biometry.")

    @st.cache_resource
    def load_model():
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = smp.Unet(encoder_name="resnet34", encoder_weights=None, in_channels=3, classes=1)
        model.load_state_dict(torch.load("afca_unet_model.pth", map_location=device))
        model.eval()
        return model, device

    model, device = load_model()
    PIXEL_TO_MM = 0.25 

    uploaded_file = st.file_uploader("Upload Ultrasound Image (.jpg, .png)", type=["jpg", "png", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        raw_image = np.array(image)
        
        with st.spinner("Analyzing scan and computing biometry..."):
            resized_image = cv2.resize(raw_image, (256, 256))
            image_tensor = resized_image / 255.0
            image_tensor = np.transpose(image_tensor, (2, 0, 1))
            image_tensor = torch.tensor(image_tensor, dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                prediction = model(image_tensor)
                prediction = torch.sigmoid(prediction)
                predicted_mask = (prediction.squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            output_image = resized_image.copy()
            contours, _ = cv2.findContours(predicted_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            mature_count = 0
            total_count = 0
            sizes_list = [] 

            for contour in contours:
                area_px = cv2.contourArea(contour)
                if area_px < 50: 
                    continue
                    
                total_count += 1
                radius_px = math.sqrt(area_px / math.pi)
                diameter_mm = (radius_px * 2) * PIXEL_TO_MM
                sizes_list.append(round(diameter_mm, 1))
                
                if diameter_mm >= 18.0:
                    color = (0, 255, 0) 
                    mature_count += 1
                elif diameter_mm >= 10.0:
                    color = (255, 255, 0) 
                else:
                    color = (255, 0, 0) 
                    
                cv2.drawContours(output_image, [contour], -1, color, 2)
                
                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.putText(output_image, f"{diameter_mm:.1f}mm", (cX - 15, cY), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Scan")
            st.image(raw_image, use_container_width=True)
        with col2:
            st.subheader("AFCA Automated Biometry")
            st.image(output_image, use_container_width=True)
            
        st.success(f"**Clinical Summary:** Detected {total_count} total follicles. {mature_count} follicles are >= 18mm.")

        # --- PDF Generation Feature ---
        st.write("---")
        st.subheader("Generate Clinical Report")
        
        sizes_list.sort(reverse=True) 
        
        def create_pdf():
            pdf = FPDF()
            pdf.add_page()
            
            pdf.set_font("Arial", 'B', 16)
            pdf.cell(200, 10, txt="AFCA - Automated Clinical Ultrasound Report", ln=True, align='C')
            pdf.line(10, 20, 200, 20)
            pdf.ln(10)
            
            pdf.set_font("Arial", size=12)
            # Pulls the actual username of the doctor who logged in!
            pdf.cell(200, 8, txt=f"Attending Physician: Dr. {st.session_state['username']}", ln=True)
            pdf.cell(200, 8, txt="Patient ID: P-84920", ln=True)
            pdf.cell(200, 8, txt="Scan Type: Transvaginal Pelvic Ultrasound", ln=True)
            pdf.ln(10)
            
            pdf.set_font("Arial", 'B', 14)
            pdf.cell(200, 10, txt="AI Biometry Results:", ln=True)
            pdf.set_font("Arial", size=12)
            pdf.cell(200, 8, txt=f"Total Fluid Pockets Detected: {total_count}", ln=True)
            pdf.cell(200, 8, txt=f"Mature Follicles (Ready for Trigger): {mature_count}", ln=True)
            pdf.ln(5)
            
            pdf.cell(200, 8, txt="Measurements (Largest to Smallest):", ln=True)
            for i, size in enumerate(sizes_list):
                status = "MATURE" if size >= 18.0 else "Growing"
                pdf.cell(200, 8, txt=f"  - Follicle {i+1}: {size} mm [{status}]", ln=True)
            
            temp_pdf_path = tempfile.mktemp(suffix=".pdf")
            pdf.output(temp_pdf_path)
            return temp_pdf_path

        pdf_file_path = create_pdf()
        
        with open(pdf_file_path, "rb") as pdf_file:
            st.download_button(
                label="📥 Download PDF Clinical Report",
                data=pdf_file,
                file_name="AFCA_Patient_Report.pdf",
                mime="application/pdf"
            )
