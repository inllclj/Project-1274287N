import streamlit as st
import tensorflow as tf
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input

# 1. LOAD MODEL (Code 2 logic, but optimized)
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('best_model.h5')

model = load_model()

# 2. THE PREDICTION FUNCTION (Code 2 logic)
def predict(image):
    # Process image (resize, array, preprocess_input)
    # ...
    prediction = model.predict(img_array)
    return "Faulty" if prediction[0][0] > 0.5 else "OK"

# 3. STREAMLIT UI
st.title("Casting Defect Scanner")
uploaded_file = st.file_uploader("Upload Image")
if uploaded_file:
    result = predict(uploaded_file)
    st.write(f"The part is: {result}")

----------------------------------------------------------------------------------------------------
Gemini said
Yes, you absolutely can! Combining them into one UI is a great way to compare how the two different frameworks (PyTorch vs. Keras) perform on the same image.

To do this, we will use a Streamlit Sidebar to let the user choose which "engine" they want to use for the inspection.

The "Dual-Engine" app.py
Python
import streamlit as st
from PIL import Image
import numpy as np

# --- PAGE SETUP ---
st.set_page_config(page_title="Dual-Engine Casting Inspector", layout="wide")
st.title("🏭 Universal Casting Defect Detector")
st.write("Compare PyTorch and Keras model predictions in one place.")

# --- SIDEBAR SELECTION ---
st.sidebar.header("Settings")
model_type = st.sidebar.selectbox("Select Model Engine", ("PyTorch (Custom CNN)", "Keras (VGG16)"))

# --- 1. LOAD MODELS (Cached) ---
@st.cache_resource
def load_keras():
    import tensorflow as tf
    return tf.keras.models.load_model('best_casting_model.h5')

@st.cache_resource
def load_pytorch():
    import torch
    # Note: Ensure your SimpleCNN class definition is included here!
    model = SimpleCNN() 
    model.load_state_dict(torch.load('best_casting_model.pth', map_location='cpu'))
    model.eval()
    return model

# --- 2. PREPROCESSING FUNCTIONS ---
def predict_keras(model, image):
    from tensorflow.keras.applications.vgg16 import preprocess_input
    img = image.resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    pred = model.predict(img_array)
    return pred[0][0]

def predict_pytorch(model, image):
    from torchvision import transforms
    import torch
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(img_tensor)
    return output.item()

# --- 3. MAIN UI LOGIC ---
uploaded_file = st.file_uploader("Upload casting image...", type=["jpg", "jpeg", "png"])

if uploaded_file:
    col1, col2 = st.columns(2)
    image = Image.open(uploaded_file).convert('RGB')
    
    with col1:
        st.image(image, caption="Uploaded Image", use_container_width=True)
    
    with col2:
        st.subheader(f"Results: {model_type}")
        with st.spinner('Analyzing...'):
            if model_type == "Keras (VGG16)":
                model = load_keras()
                score = predict_keras(model, image)
            else:
                model = load_pytorch()
                score = predict_pytorch(model, image)
            
            # Display Result
            if score > 0.5:
                st.error(f"⚠️ FAULTY (Confidence: {score:.2%})")
            else:
                st.success(f"✅ OK (Confidence: {1-score:.2%})")