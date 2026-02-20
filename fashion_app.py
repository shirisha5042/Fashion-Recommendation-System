import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import zipfile
import requests
import gdown

st.set_page_config(
    page_title="Fashion Recommendation System",
    layout="wide"
)

st.markdown("""
<style>
    .stButton>button {
        background: linear-gradient(90deg, #FF69B4, #FF1493);
        color: white;
        font-weight: bold;
        border-radius: 20px;
        padding: 0.5rem 2rem;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Google Drive Configuration
GOOGLE_DRIVE_FILE_ID = "18BZUrFg6aY5sujhWlhVsUUtDr0Q24SEP"  # Your mytra.zip file on Google Drive
GOOGLE_DRIVE_ZIP_URL = f"https://drive.google.com/uc?export=download&id={GOOGLE_DRIVE_FILE_ID}"

@st.cache_resource
def setup_dataset():
    """
    Downloads and extracts the dataset from Google Drive (only on first run).
    Returns the path to the extracted dataset folder.
    """
    zip_path = "myntradataset.zip"
    dataset_folder = "myntradataset"  # This matches your Google Drive folder name
    
    # Check if dataset already exists
    if os.path.exists(dataset_folder) and os.path.exists(f"{dataset_folder}/images"):
        st.sidebar.info("✅ Dataset already loaded")
        return dataset_folder
    
    # Download ZIP from Google Drive
    if not os.path.exists(zip_path):
        try:
            with st.spinner("📥 Downloading dataset from Google Drive (3GB - first time only)..."):
                # Using gdown for better Google Drive support
                gdown.download(GOOGLE_DRIVE_ZIP_URL, zip_path, quiet=False)
                st.sidebar.success("✅ Dataset downloaded successfully!")
        except Exception as e:
            st.error(f"❌ Error downloading dataset: {str(e)}")
            st.info("💡 Make sure your Google Drive link is public and the FILE_ID is correct")
            st.stop()
    
    # Extract ZIP
    if not os.path.exists(dataset_folder):
        try:
            with st.spinner("📂 Extracting dataset..."):
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall()
                st.sidebar.success("✅ Dataset extracted successfully!")
        except Exception as e:
            st.error(f"❌ Error extracting dataset: {str(e)}")
            st.stop()
    
    # Verify dataset structure
    if os.path.exists(f"{dataset_folder}/images"):
        image_count = len([f for f in os.listdir(f"{dataset_folder}/images") if f.endswith(('.jpg', '.jpeg', '.png'))])
        st.sidebar.success(f"✅ Dataset ready! {image_count:,} images found")
    else:
        st.warning("⚠️ Dataset structure might be incorrect. Expected: mytradataset/images/")
    
    return dataset_folder

class FeatureExtractor:
    def __init__(self, model_name='resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        if model_name == 'resnet50':
            self.model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model = self.model.to(self.device)
        self.model.eval()
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image):
        if isinstance(image, str):
            image = Image.open(image)
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_tensor = self.transform(image).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            features = self.model(image_tensor)
        
        return features.squeeze().cpu().numpy()

class FashionRecommender:
    def __init__(self, features, metadata_df):
        self.features = features
        self.metadata = metadata_df
        
        self.knn_model = NearestNeighbors(
            n_neighbors=min(20, len(features)),
            metric='cosine',
            algorithm='brute'
        )
        self.knn_model.fit(features)
    
    def get_recommendations(self, item_index, n_recommendations=6):
        query_features = self.features[item_index].reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(query_features, n_neighbors=n_recommendations+1)
        
        indices = indices[0][1:]
        distances = distances[0][1:]
        
        recommendations = self.metadata.iloc[indices].copy()
        recommendations['similarity_score'] = 1 - distances
        
        return recommendations
    
    def find_similar_to_uploaded(self, uploaded_features, n_recommendations=6):
        uploaded_features = uploaded_features.reshape(1, -1)
        distances, indices = self.knn_model.kneighbors(uploaded_features, n_neighbors=n_recommendations)
        
        indices = indices[0]
        distances = distances[0]
        
        recommendations = self.metadata.iloc[indices].copy()
        recommendations['similarity_score'] = 1 - distances
        
        return recommendations

@st.cache_resource
def load_model():
    if not os.path.exists('models/fashion_recommender.pkl'):
        st.error("Model not found! Please train the model first.")
        st.code("python train_myntra_model.py")
        st.stop()
    
    with open('models/fashion_recommender.pkl', 'rb') as f:
        model_data = joblib.load(f)
    
    features = model_data['features']
    metadata = model_data['metadata']
    
    # Fix Windows paths to work on Linux (Streamlit Cloud)
    if 'image_path' in metadata.columns:
        metadata['image_path'] = metadata['image_path'].str.replace('\\', '/', regex=False)
    
    st.sidebar.success("Model Loaded!")
    st.sidebar.info(f"Items: {len(metadata):,}")
    
    recommender = FashionRecommender(features, metadata)
    extractor = FeatureExtractor('resnet18')
    
    return recommender, extractor, metadata

def browse_catalog_mode(recommender, metadata):
    st.header("Browse Fashion Catalog")
    
    col1, col2 = st.columns([2, 1])
    with col1:
        st.write(f"**Total Items:** {len(metadata):,}")
    with col2:
        n_recommendations = st.slider("Recommendations", 3, 9, 6)
    
    st.subheader("Select an Item")
    
    if 'sample_items' not in st.session_state or st.button("Shuffle"):
        st.session_state.sample_items = metadata.sample(min(10, len(metadata)))
    
    sample_items = st.session_state.sample_items
    
    cols = st.columns(5)
    selected_idx = None
    
    for idx, (_, item) in enumerate(sample_items.iterrows()):
        col = cols[idx % 5]
        with col:
            try:
                img = Image.open(item['image_path'])
                st.image(img, use_container_width=True)
                if st.button("Select", key=f"btn_{idx}"):
                    selected_idx = item.name
            except:
                st.write("Error")
    
    if selected_idx is not None:
        st.markdown("---")
        st.subheader("Recommendations")
        
        query_item = metadata.iloc[selected_idx]
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.markdown("### Query Item")
            try:
                query_img = Image.open(query_item['image_path'])
                st.image(query_img, use_container_width=True)
                st.caption(query_item['filename'][:30])
            except:
                st.error("Error loading")
        
        with col2:
            recommendations = recommender.get_recommendations(selected_idx, n_recommendations)
            
            if len(recommendations) > 0:
                cols = st.columns(3)
                for idx, (_, rec) in enumerate(recommendations.iterrows()):
                    col = cols[idx % 3]
                    with col:
                        try:
                            rec_img = Image.open(rec['image_path'])
                            st.image(rec_img, use_container_width=True)
                            st.write(f"**{rec['similarity_score']:.1%}**")
                            st.caption(rec['filename'][:20])
                        except:
                            st.write("Error")

def upload_image_mode(recommender, extractor, metadata):
    st.header("Upload Your Image")
    
    uploaded_file = st.file_uploader("Choose an image", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.subheader("Your Image")
            st.image(image, use_container_width=True)
        
        with col2:
            st.subheader("Similar Items")
            n_recommendations = st.slider("Results", 3, 12, 6)
            
            with st.spinner("Analyzing..."):
                features = extractor.extract_features(image)
            
            recommendations = recommender.find_similar_to_uploaded(features, n_recommendations)
            
            cols = st.columns(3)
            for idx, (_, rec) in enumerate(recommendations.iterrows()):
                col = cols[idx % 3]
                with col:
                    try:
                        rec_img = Image.open(rec['image_path'])
                        st.image(rec_img, use_container_width=True)
                        st.write(f"**{rec['similarity_score']:.1%}**")
                        st.caption(rec['filename'][:25])
                    except:
                        st.write("Error")

def analytics_dashboard(metadata):
    st.header("Dataset Analytics")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Items", f"{len(metadata):,}")
    with col2:
        st.metric("Unique Images", len(metadata['image_id'].unique()))
    with col3:
        st.metric("Features", "512D")
    
    st.markdown("---")
    st.subheader("Sample Images")
    
    cols = st.columns(6)
    sample_items = metadata.sample(min(12, len(metadata)))
    
    for idx, (_, item) in enumerate(sample_items.iterrows()):
        col = cols[idx % 6]
        with col:
            try:
                img = Image.open(item['image_path'])
                st.image(img, use_container_width=True)
            except:
                pass
    
    st.markdown("---")
    st.subheader("Dataset Info")
    st.dataframe(metadata.head(20), use_container_width=True)

def main():
    st.markdown('<h1 style="text-align: center;">Fashion Recommendation System </h1>', unsafe_allow_html=True)
    
    # Setup dataset from Google Drive (downloads on first run only)
    dataset_path = setup_dataset()
    
    recommender, extractor, metadata = load_model()
    
    st.sidebar.title("Navigation")
    mode = st.sidebar.radio("Select Mode", ["Browse Catalog", "Upload Image", "Analytics"], label_visibility="collapsed")
    
    if mode == "Browse Catalog":
        browse_catalog_mode(recommender, metadata)
    elif mode == "Upload Image":
        upload_image_mode(recommender, extractor, metadata)
    else:
        analytics_dashboard(metadata)

if __name__ == "__main__":
    main()
