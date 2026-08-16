👗 Fashion Recommendation System

An AI-powered Fashion Recommendation System that recommends visually similar fashion products based on an uploaded image or a selected product from the catalog.

The project uses transfer learning with ResNet50 to extract deep visual features from fashion images and K-Nearest Neighbors with a Ball Tree index to efficiently retrieve visually similar products.

🔗 Live Demo: https://fashionrecommendationsystem.streamlit.app/

---

📌 Project Overview

Online fashion catalogs contain thousands of products, making it difficult for users to discover items visually similar to something they already like.

This project addresses this problem by allowing users to:

- Browse available fashion products
- Upload a fashion image
- Extract visual features from the image using a pretrained CNN
- Find visually similar products
- View recommendations through an interactive Streamlit interface
- Explore basic analytics and dataset information

The system focuses on content-based visual recommendation, meaning recommendations are generated from the visual characteristics of the products rather than user ratings or purchase history.

---

🎯 Objectives

- Build an image-based fashion recommendation system using deep learning.
- Extract meaningful visual representations from fashion images.
- Efficiently retrieve visually similar products from a large dataset.
- Provide an easy-to-use web interface for fashion discovery.
- Evaluate the recommendation performance using standard information-retrieval metrics.

---

🧠 Methodology

The overall pipeline consists of the following stages:

Fashion Dataset
       ↓
Image Preprocessing
       ↓
Pretrained ResNet50
       ↓
2048-D Feature Extraction
       ↓
L2 Normalization
       ↓
Ball Tree / KNN Similarity Search
       ↓
Top-K Similar Fashion Products
       ↓
Streamlit Web Application

1. Image Preprocessing

Fashion product images are resized to 224 × 224 pixels, matching the input requirements of ResNet50.

2. Feature Extraction

A pretrained ResNet50 model with ImageNet weights is used as the feature extractor.

The classification layer is removed and the network is used to generate a 2048-dimensional visual feature vector for each image.

3. Feature Normalization

The extracted feature vectors are L2-normalized before similarity search.

This allows the system to compare images based on their learned visual representations.

4. Similarity Search

A Ball Tree-based KNN index is constructed over the normalized feature vectors.

For a query image, the system searches the index and retrieves the most visually similar fashion products.

5. Web Application

The recommendation system is integrated into a Streamlit application that provides an interactive interface for exploring the fashion catalog and generating recommendations.

---

🏗️ Technology Stack

Category| Technologies
Programming Language| Python
Deep Learning| TensorFlow / Keras
CNN Architecture| ResNet50
Machine Learning| Scikit-learn
Data Processing| Pandas, NumPy
Visualization| Matplotlib
Web Framework| Streamlit
Development| Google Colab, Jupyter Notebook, VS Code
Version Control| Git / GitHub

---

📊 Model Comparison

Different pretrained CNN architectures were considered during development.

Model| Feature Dimension| Observation
ResNet18| 512| Lightweight feature representation
VGG16| 4096| High-dimensional representation
ResNet50| 2048| Good balance of representation quality and computational efficiency

ResNet50 was selected as the primary feature extractor because it provided a suitable balance between feature representation, retrieval performance, and computational requirements.

---

📈 Results

The developed system achieved the following results during evaluation:

Metric| Score
Accuracy| 84.7%
Precision@6| 82.3%
Recall@6| 79.1%
F1-Score@6| 80.7%
Top-1 Similarity| 93.4%

These results demonstrate the effectiveness of deep visual embeddings for content-based fashion recommendation.

---

🖥️ Application Features

🛍️ Browse Catalog

Explore products available in the fashion dataset.

📷 Upload & Discover

Upload a fashion image and receive visually similar product recommendations.

📊 Analytics Dashboard

View relevant dataset statistics and visual analytics.

---

📂 Repository Structure

Fashion-Recommendation-System/
│
├── .streamlit/
├── models/
├── Fashion_Final_Notebook.ipynb
├── Fashion_IEEE_Paper_Revised.tex
├── fashion_app.py
├── packages.txt
├── requirements.txt
├── .gitignore
└── README.md

---

🚀 Running the Project Locally

1. Clone the repository

git clone https://github.com/shirisha5042/Fashion-Recommendation-System.git
cd Fashion-Recommendation-System

2. Install dependencies

pip install -r requirements.txt

3. Run the Streamlit application

streamlit run fashion_app.py

The application will open in your browser.

---

🌐 Live Demo

Try the deployed application:

https://fashionrecommendationsystem.streamlit.app/

---

👥 Team

This project was developed as a 4-member final-year academic project.

My Contribution

I worked as a core developer and was primarily responsible for:

- Machine learning and computer vision pipeline
- ResNet50 feature extraction
- Image preprocessing
- Feature normalization
- KNN/Ball Tree similarity retrieval
- Model comparison and evaluation
- Streamlit application development
- Integration and testing

The project was developed collaboratively, with team members supporting different areas of the system.

---

🔮 Future Scope

- Incorporate user preferences and purchase history for personalized recommendations.
- Explore Vision Transformers and newer visual embedding models.
- Improve scalability for very large product catalogs.
- Add category, color, style, and brand-aware filtering.
- Develop a hybrid recommendation system combining visual similarity with user behavior.
- Deploy the system using scalable cloud infrastructure.

---

📄 Project Documentation

The repository also contains the project notebook and IEEE paper source used during the academic project.

---

⭐ Conclusion

The Fashion Recommendation System demonstrates how deep learning, transfer learning, and similarity-based retrieval can be combined to build an interactive visual fashion discovery platform.

The project provides a practical implementation of computer vision and recommendation-system techniques and can serve as a foundation for more personalized and scalable fashion recommendation systems.
