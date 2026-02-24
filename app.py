import streamlit as st
import numpy as np
from tensorflow.keras.applications.vgg16 import preprocess_input
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms

# --- define SimpleCNN class ---
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        def convBlock(ni, no):
            return nn.Sequential(
                nn.Conv2d(ni, no, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.BatchNorm2d(no),
                nn.MaxPool2d(2),
            )

        self.features = nn.Sequential(
            convBlock(1, 16),  # Note: '1' means Grayscale
            convBlock(16, 32),
            convBlock(32, 64),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64*28*28, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
        self.loss_fn = nn.BCELoss()

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

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
    model = SimpleCNN() 
    # Load weights
    state_dict = torch.load('best_casting_model.pth', map_location='cpu')
    model.load_state_dict(state_dict)
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
        transforms.Grayscale(num_output_channels=1), # IMPORTANT: Match the '1' in your model
        transforms.Resize((224, 224)), # Assumes 224x224 based on 64*28*28 math
        transforms.ToTensor(),
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
