import streamlit as st
from PIL import Image
import torch
from src import config
from src.inference import load_generative_model, generate_caption

# Page configuration for a centered, clean layout
st.set_page_config(
    page_title="AI Image Captioner",
    page_icon="📸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for a beautiful dark-mode glassmorphic interface
st.markdown("""
<style>
/* Reset and background */
.stApp {
    background-color: #09090F;
    background-image: radial-gradient(circle at 50% 50%, #15122E 0%, #09090F 100%);
    color: #E2E8F0;
}

/* Hide default streamlit clutter */
header {visibility: hidden;}
footer {visibility: hidden;}
.viewerBadge_container__17x7G {display: none !important;}

/* Title styling */
.title-container {
    text-align: center;
    padding-top: 1.5rem;
    padding-bottom: 1.5rem;
}
.app-title {
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #C084FC 0%, #6366F1 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin-bottom: 0.1rem;
    letter-spacing: -0.5px;
}
.app-desc {
    font-size: 0.95rem;
    color: #94A3B8;
}

/* Main card */
.glass-container {
    background: rgba(255, 255, 255, 0.02);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 20px;
    padding: 2rem;
    box-shadow: 0 15px 35px rgba(0, 0, 0, 0.4);
    margin-bottom: 2rem;
}

/* Style file uploader area */
.stFileUploader section {
    background-color: rgba(255, 255, 255, 0.01) !important;
    border: 1px dashed rgba(255, 255, 255, 0.12) !important;
    border-radius: 12px !important;
    padding: 1.5rem !important;
}

/* Styled Button */
div.stButton > button {
    background: linear-gradient(135deg, #A855F7 0%, #6366F1 100%) !important;
    color: #FFFFFF !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    padding: 0.8rem 2.5rem !important;
    border-radius: 12px !important;
    border: none !important;
    box-shadow: 0 6px 20px rgba(168, 85, 247, 0.3) !important;
    transition: all 0.3s ease !important;
    width: 100% !important;
    cursor: pointer !important;
}
div.stButton > button:hover {
    box-shadow: 0 8px 25px rgba(168, 85, 247, 0.5) !important;
    transform: translateY(-2px) !important;
    border: none !important;
    color: #FFFFFF !important;
}
div.stButton > button:active {
    transform: translateY(0) !important;
}

/* Output Card styling */
.caption-card {
    background: rgba(168, 85, 247, 0.08);
    border: 1px solid rgba(168, 85, 247, 0.25);
    border-radius: 16px;
    padding: 1.5rem;
    box-shadow: 0 0 25px rgba(168, 85, 247, 0.15);
    margin-top: 1.5rem;
    text-align: center;
}
.caption-header {
    font-size: 0.8rem;
    text-transform: uppercase;
    letter-spacing: 2px;
    color: #C084FC;
    font-weight: 700;
    margin-bottom: 0.6rem;
}
.caption-text {
    font-size: 1.4rem;
    font-weight: 600;
    color: #F8FAFC;
    line-height: 1.4;
    text-shadow: 0 2px 10px rgba(168, 85, 247, 0.2);
}
</style>
""", unsafe_allow_html=True)

# ----------------- CACHED BLIP LOAD -----------------
@st.cache_resource
def load_blip():
    return load_generative_model("Salesforce/blip-image-captioning-base", config.DEVICE)

# ----------------- APP LAYOUT -----------------

st.markdown("""
<div class="title-container">
    <div class="app-title">📸 AI Image Captioner</div>
    <div class="app-desc">Upload an image to generate professional, descriptive captions.</div>
</div>
""", unsafe_allow_html=True)

# Pre-load BLIP model in background
with st.spinner("Initializing generative captioning network..."):
    blip_model, blip_processor = load_blip()

# Initialize session state to persist generated caption between button runs
if "caption_result" not in st.session_state:
    st.session_state.caption_result = ""
if "last_uploaded_file" not in st.session_state:
    st.session_state.last_uploaded_file = None

st.markdown('<div class="glass-container">', unsafe_allow_html=True)

# 1. Image Uploader
uploaded_file = st.file_uploader(
    "Choose an image...", 
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# Reset session state if a new file is uploaded
if uploaded_file != st.session_state.last_uploaded_file:
    st.session_state.caption_result = ""
    st.session_state.last_uploaded_file = uploaded_file

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Display image with container width to fix deprecation warning
    st.image(image, use_container_width=True)
    
    st.markdown("<div style='height: 15px;'></div>", unsafe_allow_html=True)
    
    # 2. Button of generate caption
    if st.button("✨ Generate Caption"):
        with st.spinner("Extracting features and generating description..."):
            caption = generate_caption(image, blip_model, blip_processor, config.DEVICE)
            st.session_state.caption_result = caption

# 3. Caption display
if st.session_state.caption_result:
    st.markdown(f"""
    <div class="caption-card">
        <div class="caption-header">🔮 GENERATED DESCRIPTION</div>
        <div class="caption-text">"{st.session_state.caption_result}"</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)
