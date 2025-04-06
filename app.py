import streamlit as st

# This MUST be the first Streamlit command
st.set_page_config(page_title="Retina Disease Detection", page_icon="👁️", layout="wide")

# Now import all other libraries
import numpy as np
import cv2
import sys
import tensorflow as tf
import os
import time
import gdown
import pathlib

# Debug information
st.write(f"Python version: {sys.version}")
st.write(f"Python executable: {sys.executable}")
st.write(f"Current working directory: {os.getcwd()}")

# First, properly import vit-keras to handle custom layers
try:
    # Import the vit_keras package directly
    from vit_keras import vit
    st.sidebar.success("✅ vit-keras package loaded successfully")
except ImportError:
    st.error("vit-keras package not installed. Please run: pip install vit-keras tensorflow-addons")
    st.stop()

# Configuration
IMG_SIZE = (224, 224)
CLASSES = {
    0: 'normal',
    1: 'glaucoma',
    2: 'dr_mild',
    3: 'dr_moderate',
    4: 'dr_severe',
    5: 'dr_proliferative'
}

# Function to download model from Google Drive
def download_model_from_gdrive(model_url, output_path):
    try:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Download the file if it doesn't exist
        if not os.path.exists(output_path):
            with st.spinner(f"Downloading model from Google Drive... This may take a while."):
                gdown.download(model_url, output_path, quiet=False)
            st.success(f"✅ Model downloaded successfully to {output_path}!")
        else:
            st.success(f"✅ Model already exists at {output_path}!")
        
        return True
    except Exception as e:
        st.error(f"Error downloading model: {str(e)}")
        return False

def preprocess_image(uploaded_file, green_channel_only=True):
    # Read image from uploaded file
    img = np.frombuffer(uploaded_file.read(), np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    
    # Create a copy for display
    display_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Preprocess for model
    if green_channel_only:
        green_channel = img[:, :, 1]
        clahe_op = cv2.createCLAHE(clipLimit=2)
        enhanced_green = clahe_op.apply(green_channel)
        processed_img = cv2.cvtColor(enhanced_green, cv2.COLOR_GRAY2RGB)
    else:
        processed_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
    processed_img = cv2.resize(processed_img, IMG_SIZE)
    
    # Normalize image for model input
    processed_img = processed_img / 255.0
    processed_img = np.expand_dims(processed_img, axis=0)
    
    return processed_img, display_img

def predict_with_model(model, img):
    try:
        # Make prediction
        predictions = model.predict(img)
        
        # Convert to dictionary with class names and probabilities
        results = {}
        for i, class_name in CLASSES.items():
            results[class_name] = float(predictions[0][i] * 100)
        
        # Sort by probability (descending)
        sorted_results = {k: v for k, v in sorted(results.items(), key=lambda item: item[1], reverse=True)}
        
        return sorted_results, None
    except Exception as e:
        return None, f"Error making prediction: {str(e)}"

def get_condition_info(condition):
    info = {
        'normal': """
            Your retinal scan appears normal. No signs of diabetic retinopathy or glaucoma were detected.
            
            **Recommendation**: Continue with regular eye check-ups as recommended by your healthcare provider.
        """,
        'glaucoma': """
            **Glaucoma** is a group of eye conditions that damage the optic nerve, often caused by abnormally high pressure in your eye.
            
            **Key facts**:
            - Often called the "silent thief of sight" as it can develop without noticeable symptoms
            - Early detection and treatment can help prevent vision loss
            - Regular eye pressure checks are important for monitoring
            
            **Recommendation**: Consult with an ophthalmologist for a comprehensive examination and to discuss treatment options.
        """,
        'dr_mild': """
            **Mild Diabetic Retinopathy** is the early stage of the condition where small areas of balloon-like swelling occur in the retina's tiny blood vessels.
            
            **Key facts**:
            - May not cause noticeable vision changes at this stage
            - Regular monitoring is essential to prevent progression
            - Controlling blood sugar, blood pressure, and cholesterol can slow progression
            
            **Recommendation**: Schedule an appointment with an ophthalmologist and ensure your diabetes management is optimal.
        """,
        'dr_moderate': """
            **Moderate Diabetic Retinopathy** occurs as the disease progresses, with more blood vessels becoming blocked.
            
            **Key facts**:
            - May start affecting vision
            - Blood vessels to the retina begin to weaken and leak
            - Careful monitoring and treatment is important
            
            **Recommendation**: Consult with a retina specialist promptly for potential treatment options.
        """,
        'dr_severe': """
            **Severe Diabetic Retinopathy** is characterized by significant blockage of blood vessels and signals the retina to grow new blood vessels.
            
            **Key facts**:
            - Vision may be noticeably affected
            - High risk of progressing to the proliferative stage
            - Immediate medical attention is required
            
            **Recommendation**: Seek immediate care from a retina specialist or ophthalmologist.
        """,
        'dr_proliferative': """
            **Proliferative Diabetic Retinopathy** is the advanced stage where new blood vessels grow on the surface of the retina. These vessels are fragile and can leak, causing severe vision problems.
            
            **Key facts**:
            - High risk of vision loss
            - May require laser treatment or surgery
            - Medical emergency requiring specialist care
            
            **Recommendation**: Seek immediate care from a retina specialist. This condition may require urgent intervention.
        """
    }
    return info.get(condition, "Information not available.")

# Custom CSS
# Custom CSS - Updated with better color contrast
st.markdown("""
    <style>
    .main {
        background-color: #0E1117;
        color: #FFFFFF;
    }
    [data-testid="stAppViewContainer"] {
        background-color: #0E1117;
    }
    [data-testid="stHeader"] {
        background-color: #0E1117;
    }
    [data-testid="stToolbar"] {
        display: none;
    }
    .stProgress > div > div > div > div {
        background-color: #00CC66;
    }
    .prediction-box {
        background-color: #1E2530;
        border-radius: 10px;
        padding: 20px;
        margin: 10px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.2);
        color: #FFFFFF;
    }
    .stDeployButton {
        display: none;
    }
    footer {
        display: none;
    }
    [data-testid="stSidebar"] {
        background-color: #0E1117;
    }
    .uploadedFile {
        background-color: #1E2530 !important;
        border: 1px solid #2E3440 !important;
        color: #FFFFFF !important;
    }
    .css-1dp5vir {
        background-color: #1E2530 !important;
        border: 1px solid #2E3440 !important;
        color: #FFFFFF !important;
    }
    .css-1n76uvr {
        color: #FFFFFF !important;
    }
    .stAlert {
        background-color: #1E2530 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.freepik.com/free-psd/iris-eye-isolated_23-2151866148.jpg", width=100)
    st.title("Retina Analysis")
    st.markdown("---")
    
    st.subheader("About")
    st.write("""
        This application uses deep learning to detect potential eye conditions from retinal images.
        
        The model utilizes a hybrid Vision Transformer + ResNet architecture with 98% accuracy.
    """)
    
    st.markdown("---")
    
    st.subheader("Settings")
    green_channel = st.checkbox("Using Green Channel Enhancement", value=True, 
                             help="Enhances the green channel which often shows better contrast for retinal features")
    
    # Google Drive model settings
    st.subheader("Model Settings")
    model_url = st.text_input(
        "Google Drive Model URL", 
        value="https://drive.google.com/file/d/10KCwLSCWsnzX2Fauk0Hj23_wtAsLxTiA/view?usp=drive_link",
        help="URL to your model on Google Drive"
    )
    
    # Convert the Google Drive URL to a direct download link
    if "drive.google.com" in model_url and "file/d/" in model_url:
        file_id = model_url.split("file/d/")[1].split("/")[0]
        direct_url = f"https://drive.google.com/uc?id={file_id}"
    else:
        direct_url = model_url
    
    # Set model path
    model_directory = "downloaded_models"
    model_filename = "retina_model.h5"
    model_path = os.path.join(model_directory, model_filename)
    
    use_mock = st.checkbox("Use mock predictions", value=False,
                          help="Enable this for demonstration if model loading fails")
    
    # Download model button
    if st.button("Download/Update Model"):
        download_success = download_model_from_gdrive(direct_url, model_path)
        if download_success and "model" in st.session_state:
            # Clear the current model to force reloading
            del st.session_state.model
    
    st.markdown("---")
    
    st.info("""
        **Disclaimer**: This tool is for educational purposes only and not a substitute for professional medical advice.
        Always consult healthcare professionals for diagnosis and treatment.
    """)

# Main content
st.title("👁️ Retina Disease Detection")
st.markdown("""
    <div style='text-align: center; margin-bottom: 20px;'>
        <p style='font-size: 18px; color: #FFFFFF;'>
            Upload a retina image to detect potential eye conditions
        </p>
    </div>
""", unsafe_allow_html=True)

# Auto-download model on first run if it doesn't exist
if not os.path.exists(model_path) and not use_mock:
    st.info(f"Model not found at {model_path}. Attempting to download...")
    download_model_from_gdrive(direct_url, model_path)

# Try to load the model
if not use_mock and "model" not in st.session_state:
    with st.spinner("Loading model - this may take a moment..."):
        try:
            # Register GELU activation if used in your model
            tf.keras.utils.get_custom_objects()['gelu'] = tf.keras.activations.gelu
            
            # Check if model file exists
            if os.path.exists(model_path):
                # Load the model - vit_keras will handle the custom layers automatically
                st.session_state.model = tf.keras.models.load_model(model_path)
                st.success(f"✅ Model loaded successfully from {model_path}!")
            else:
                st.warning(f"Model file not found at {model_path}. Please download the model first.")
                use_mock = True
                st.session_state.model = None
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.warning("Enabling mock predictions to demonstrate the UI. Check the model path and file.")
            use_mock = True
            st.session_state.model = None

# File uploader
uploaded_file = st.file_uploader("Choose a Retinal Image", type=["jpg", "jpeg", "png"], 
                                help="Upload a clear image of the retina for analysis")

# Process uploaded image
if uploaded_file is not None:
    # Display original image and processed version
    col1, col2 = st.columns([1, 1])
    
    with col1:
        # Create a spinner while processing
        with st.spinner('Processing image...'):
            # Preprocess image
            processed_img, display_img = preprocess_image(uploaded_file, green_channel_only=green_channel)
            
            # Display tabs for original and processed views
            tab1, tab2 = st.tabs(["Original Image", "Processed Image"])
            
            with tab1:
                st.image(display_img, caption='Original Uploaded Image', use_column_width=True)
            
            with tab2:
                processed_display = (processed_img[0] * 255).astype(np.uint8)
                st.image(processed_display, caption='Processed Image (Model Input)', use_column_width=True)
    
    # Make prediction
    with col2:
        with st.spinner('Analyzing image...'):
            # If model loaded correctly and not using mock data, use it for predictions
            if not use_mock and 'model' in st.session_state and st.session_state.model is not None:
                predictions, error = predict_with_model(st.session_state.model, processed_img)
                
                if error:
                    st.error(error)
                    st.warning("Using mock predictions instead")
                    # Fall back to mock predictions
                    mock_preds = {
                        'normal': 75.2,
                        'glaucoma': 12.5,
                        'dr_mild': 6.8,
                        'dr_moderate': 3.1,
                        'dr_severe': 1.4,
                        'dr_proliferative': 1.0
                    }
                    predictions = mock_preds
            else:
                # Mock predictions for demonstration
                st.info("Using mock predictions for demonstration")
                mock_preds = {
                    'normal': 75.2,
                    'glaucoma': 12.5,
                    'dr_mild': 6.8,
                    'dr_moderate': 3.1,
                    'dr_severe': 1.4,
                    'dr_proliferative': 1.0
                }
                predictions = mock_preds
            
            # Main output in the second column
            st.markdown("<h3 style='text-align: center; color: #FFFFFF;'>Analysis Results</h3>", unsafe_allow_html=True)
            
            # Find the top condition
            top_condition = next(iter(predictions))
            top_probability = predictions[top_condition]
            
            # Display each prediction with appropriate styling
            for label, prob in predictions.items():
                # Set color based on probability
                # Set color based on probability with better visibility
                if prob > 50:
                    color = "#FF4B4B" if label != 'normal' else "#00CC66"  # Darker red and green for better visibility
                elif prob < 20:
                    color = "#00CC66" if label == 'normal' else "#FFA500"  # Darker green and orange
                else:
                    color = "#FFA500"  # Darker orange
                
                # Determine if this is the top prediction
                is_top = (label == top_condition)
                highlight_class = "highlight-box" if is_top else ""
                
                # Format the label for display
                display_label = label.replace('_', ' ').title()
                
                # Create the prediction box
                # Create the prediction box with improved visibility
                st.markdown(f"""
                    <div class='prediction-box {highlight_class}'>
                        <div style='display: flex; justify-content: space-between; align-items: center;'>
                            <span style='font-size: 18px; font-weight: bold; color: #FFFFFF;'>{display_label}</span>
                            <span style='font-size: 18px; color: {color};'>{prob:.2f}%</span>
                        </div>
                        <div style='margin-top: 10px;'>
                            <div style='background-color: #2E3440; height: 10px; border-radius: 5px;'>
                                <div style='background-color: {color}; width: {min(prob, 100)}%; height: 100%; border-radius: 5px;'></div>
                            </div>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Summary message based on top prediction probability
            if top_probability > 50:
                if top_condition == 'normal':
                    st.success(f"✅ High probability of normal retina ({top_probability:.2f}%).")
                else:
                    st.warning(f"⚠️ High probability of {top_condition.replace('_', ' ')} detected ({top_probability:.2f}%). Please consult an ophthalmologist.")
            elif top_probability < 20:
                st.info(f"ℹ️ Low confidence in predictions. Consider retaking the image with better quality.")
            else:
                st.info(f"ℹ️ Moderate probability of {top_condition.replace('_', ' ')} detected ({top_probability:.2f}%). Consider consulting an eye specialist.")
            
            # Information expander for the top condition
            with st.expander("More Information", expanded=True):
                st.markdown(f"### About {top_condition.replace('_', ' ').title()}")
                st.markdown(get_condition_info(top_condition))
                
                # Updated reference image display to use your local images
                # Map the predicted condition to the corresponding image filename
                image_filename_map = {
                    'normal': 'normal.jpeg',
                    'glaucoma': 'glaucoma.png',
                    'dr_mild': 'mild_dr.jpeg',
                    'dr_moderate': 'moderate_dr.jpeg',
                    'dr_severe': 'severe_dr.jpeg',
                    'dr_proliferative': 'proliferative_dr.jpeg'
                }
                
                # Always show a reference image based on the top prediction
                st.markdown("### Visual Reference")
                ref_image_path = f"reference_images/{image_filename_map.get(top_condition)}"
                
                try:
                    st.image(ref_image_path, 
                            caption=f"Example of {top_condition.replace('_', ' ').title()}", 
                            use_column_width=True)
                except Exception as e:
                    st.error(f"Could not load reference image: {str(e)}")
else:
    # Instructions when no file is uploaded
    st.info("👈 Please upload a retinal image to get started")
    
    # Example images (optional)
    with st.expander("Examples of Retinal Images"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image("reference_images/normal.jpeg", caption="Normal Retina")
        with col2:
            st.image("reference_images/mild_dr.jpeg", caption="Mild Diabetic Retinopathy")
        with col3:
            st.image("reference_images/glaucoma.png", caption="Glaucoma")

# Add information about the application
st.markdown("---")
st.markdown("""
<div style='text-align: center; margin-top: 20px;'>
    <p style='font-size: 14px; color: #FFFFFF;'>
        This application uses a hybrid Vision Transformer + ResNet architecture to analyze retinal images. 
        It can detect normal retinas, glaucoma, and various stages of diabetic retinopathy with 98% accuracy.
    </p>
</div>
""", unsafe_allow_html=True)