import streamlit as st
import torch
import torchvision
from torchvision import transforms, models
from torch import nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import os

# Page configuration
st.set_page_config(
    page_title="Vehicle Type Classification System",
    page_icon="🚗",
    layout="wide"
)

# Title
st.title("🚗 Vehicle Type Classification System")
st.markdown("### Using CNN ResNet-18 for Vehicle Recognition")

# Sidebar
st.sidebar.header("About")
st.sidebar.info(
    """
    This system classifies vehicles into:
    - 🚗 Car
    - 🏍️ Motorcycle  
    - 🚚 Truck
    """
)

# Model loading
@st.cache_resource
def load_model():
    """Load the trained ResNet-18 model"""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 3
    
    # Create model
    model = models.resnet18(pretrained=False)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    # Load weights
    model_path = 'models/resnet18_finetuned.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        st.sidebar.success("✅ Model loaded successfully!")
    else:
        st.sidebar.error(f"❌ Model file not found at {model_path}")
        st.sidebar.info("Please make sure the model file exists in the 'models' folder")
        st.stop()
    
    model.to(device)
    model.eval()
    return model, device

# Image preprocessing
def preprocess_image(image):
    """Preprocess the uploaded image"""
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    return transform(image).unsqueeze(0)

# Prediction function
def predict(model, image_tensor, device):
    """Make prediction on the image"""
    class_names = ['Car', 'Motorcycle', 'Truck']
    
    image_tensor = image_tensor.to(device)
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
        
    return class_names[predicted.item()], confidence.item(), probabilities[0].cpu()

# Main UI
st.markdown("---")
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Image")
    uploaded_file = st.file_uploader(
        "Choose a vehicle image...", 
        type=['jpg', 'jpeg', 'png']
    )
    
    if uploaded_file is not None:
        # Display image
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_container_width=True)
        
        # Classify button with unique key
        if st.button('🔍 Classify Vehicle', type='primary', key='classify_btn'):
            # Load model
            model, device = load_model()
            
            # Process and predict
            with st.spinner('Classifying...'):
                image_tensor = preprocess_image(image)
                predicted_class, confidence, probabilities = predict(model, image_tensor, device)
            
            # Save to session state
            st.session_state.prediction = {
                'class': predicted_class,
                'confidence': confidence,
                'probabilities': probabilities
            }

with col2:
    st.header("📊 Results")
    
    if 'prediction' in st.session_state:
        pred = st.session_state.prediction
        
        # Show prediction
        emoji_map = {'Car': '🚗', 'Motorcycle': '🏍️', 'Truck': '🚚'}
        st.success(f"**Predicted:** {emoji_map[pred['class']]} {pred['class']}")
        st.info(f"**Confidence:** {pred['confidence']*100:.2f}%")
        
        # Probability chart
        st.subheader("Probability Distribution")
        classes = ['Car', 'Motorcycle', 'Truck']
        probs = pred['probabilities'].numpy()
        
        # Create bar chart
        fig, ax = plt.subplots(figsize=(8, 4))
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        bars = ax.bar(classes, probs, color=colors)
        
        # Add percentage labels
        for bar, prob in zip(bars, probs):
            height = bar.get_height()
            ax.annotate(f'{prob*100:.1f}%',
                       xy=(bar.get_x() + bar.get_width()/2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom')
        
        ax.set_ylabel('Probability')
        ax.set_title('Classification Confidence')
        ax.set_ylim(0, 1.1)
        st.pyplot(fig)
        
        # Detailed results
        with st.expander("📋 Detailed Results"):
            for cls, prob in zip(classes, probs):
                st.write(f"**{cls}**: {prob*100:.2f}%")
    else:
        st.info("👆 Upload an image and click 'Classify Vehicle' to see results")

# Footer
st.markdown("---")
st.markdown("### 💡 Tips:")
col1, col2 = st.columns(2)
with col1:
    st.markdown("✅ Use clear, well-lit images")
    st.markdown("✅ Single vehicle per image")
with col2:
    st.markdown("✅ Avoid blurry photos")
    st.markdown("✅ Front/side views work best")