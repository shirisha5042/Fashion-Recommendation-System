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
from sklearn.decomposition import PCA
import gdown
import zipfile

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

st.set_page_config(
    page_title="Vogue Analytics — Fashion AI",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Google Drive file IDs for cloud deployment ───────────────────────────────
# Replace these with your actual Google Drive file IDs after uploading
GDRIVE_FILES = {
    'resnet18_pkl':  os.environ.get('GDRIVE_RESNET18_ID', ''),   # fashion_recommender_resnet18.pkl
    'resnet50_pkl':  os.environ.get('GDRIVE_RESNET50_ID', ''),   # fashion_recommender_resnet50.pkl
    'vgg16_pkl':     os.environ.get('GDRIVE_VGG16_ID', ''),      # fashion_recommender_vgg16.pkl
    'dataset_zip':   os.environ.get('GDRIVE_DATASET_ID', ''),    # myntradataset.zip (images + styles.csv)
}

@st.cache_resource
def download_from_gdrive(file_id, output_path, is_zip=False):
    """Download a file from Google Drive if it doesn't exist locally."""
    if os.path.exists(output_path):
        return True
    if not file_id:
        return False
    try:
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else '.', exist_ok=True)
        url = f'https://drive.google.com/uc?id={file_id}'
        with st.spinner(f'Downloading {os.path.basename(output_path)}... (first time only)'):
            gdown.download(url, output_path, quiet=False)
        if is_zip:
            with st.spinner('Extracting dataset...'):
                with zipfile.ZipFile(output_path, 'r') as zip_ref:
                    zip_ref.extractall('.')
                os.remove(output_path)  # Clean up zip
        return True
    except Exception as e:
        st.error(f"Download failed: {e}")
        return False

# ─── GLOBAL STYLES ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,300;1,400&family=Outfit:wght@200;300;400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    background-color: #080810 !important;
    color: #e8e0d5 !important;
}
.stApp { background: #080810 !important; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d0d18 0%, #100c18 100%) !important;
    border-right: 1px solid rgba(212,175,107,0.15) !important;
}
[data-testid="stSidebar"] * { color: #c8bfb0 !important; }

.stButton>button {
    background: linear-gradient(135deg, #c9956c 0%, #d4af6b 50%, #b8895a 100%);
    color: #080810 !important;
    font-family: 'Outfit', sans-serif;
    font-weight: 600;
    font-size: 0.8rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    border-radius: 2px;
    padding: 0.6rem 2.2rem;
    border: none;
    transition: all .3s ease;
    box-shadow: 0 4px 20px rgba(212,175,107,0.25);
}
.stButton>button:hover {
    box-shadow: 0 6px 30px rgba(212,175,107,0.45);
    transform: translateY(-1px);
}

[data-testid="stTabs"] button {
    font-family: 'Outfit', sans-serif !important;
    font-weight: 300 !important;
    letter-spacing: 2px !important;
    text-transform: uppercase !important;
    font-size: 0.72rem !important;
    color: #8a7f72 !important;
    border-bottom: 1px solid transparent !important;
    transition: all .3s !important;
}
[data-testid="stTabs"] button[aria-selected="true"] {
    color: #d4af6b !important;
    border-bottom: 1px solid #d4af6b !important;
    background: transparent !important;
}

input[type="text"], .stTextInput input {
    background: rgba(212,175,107,0.05) !important;
    border: 1px solid rgba(212,175,107,0.2) !important;
    color: #e8e0d5 !important;
    border-radius: 2px !important;
    font-family: 'Outfit', sans-serif !important;
}

[data-testid="stDataFrame"] {
    border: 1px solid rgba(212,175,107,0.12) !important;
    border-radius: 4px !important;
}

#MainMenu, footer { visibility: hidden; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #080810; }
::-webkit-scrollbar-thumb { background: #d4af6b44; border-radius: 4px; }

@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position:  200% center; }
}
@keyframes fadeUp {
    from { opacity:0; transform:translateY(20px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes pulse-border {
    0%, 100% { box-shadow: 0 0 0 0 rgba(212,175,107,0.15); }
    50%       { box-shadow: 0 0 0 6px rgba(212,175,107,0); }
}

.hero-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: clamp(3rem, 7vw, 6rem);
    font-weight: 300;
    font-style: italic;
    letter-spacing: 0.05em;
    background: linear-gradient(135deg, #d4af6b 0%, #f0d5a0 40%, #c9956c 70%, #d4af6b 100%);
    background-size: 200% auto;
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 4s linear infinite, fadeUp .8s ease forwards;
    text-align: center;
    line-height: 1;
    margin-bottom: 0.2rem;
}
.hero-sub {
    font-family: 'Outfit', sans-serif;
    font-weight: 200;
    font-size: 0.7rem;
    letter-spacing: 6px;
    text-transform: uppercase;
    color: #8a7f72;
    text-align: center;
    animation: fadeUp 1s ease .3s both;
}
.hero-divider {
    width: 60px; height: 1px;
    background: linear-gradient(90deg, transparent, #d4af6b, transparent);
    margin: 1.5rem auto;
    animation: fadeUp 1s ease .5s both;
}

.section-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.8rem;
    font-weight: 400;
    font-style: italic;
    color: #d4af6b;
    letter-spacing: 0.03em;
    margin: 2rem 0 0.3rem 0;
}
.section-rule {
    height: 1px;
    background: linear-gradient(90deg, #d4af6b55, transparent);
    margin-bottom: 1rem;
}

.kpi-card {
    background: linear-gradient(135deg, rgba(212,175,107,0.04) 0%, rgba(201,149,108,0.06) 100%);
    border: 1px solid rgba(212,175,107,0.18);
    border-radius: 3px;
    padding: 1.6rem 1.2rem;
    text-align: center;
    position: relative;
    overflow: hidden;
    animation: pulse-border 3s ease-in-out infinite, fadeUp .7s ease forwards;
    transition: transform .3s, border-color .3s;
}
.kpi-card::before {
    content: '';
    position: absolute; top:0; left:0; right:0; height:1px;
    background: linear-gradient(90deg, transparent, #d4af6b88, transparent);
}
.kpi-card:hover { transform: translateY(-3px); border-color: rgba(212,175,107,0.4); }
.kpi-number {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.4rem;
    font-weight: 600;
    background: linear-gradient(135deg, #d4af6b, #f0d5a0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    display: block;
    line-height: 1;
}
.kpi-label { font-size: 0.65rem; font-weight: 500; letter-spacing: 3px; text-transform: uppercase; color: #6a6058; margin-top: 8px; }
.kpi-sub   { font-size: 0.72rem; color: #5a5248; margin-top: 4px; }

.insight {
    background: rgba(212,175,107,0.04);
    border-left: 2px solid #d4af6b88;
    padding: 1rem 1.2rem;
    margin: 1rem 0;
    font-size: 0.85rem;
    color: #b0a898;
    font-family: 'Outfit', sans-serif;
    font-weight: 300;
}
.insight b { color: #d4af6b; font-weight: 500; }

.score-good { color: #7ecba1 !important; font-weight: 600; }
.score-mid  { color: #d4af6b !important; font-weight: 600; }
.score-low  { color: #e07070 !important; font-weight: 600; }

.method-note {
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(255,255,255,0.06);
    border-radius: 3px;
    padding: 1rem 1.2rem;
    font-size: 0.8rem;
    color: #6a6058;
    line-height: 1.6;
    margin-bottom: 1.5rem;
}
.method-note b { color: #a89880; }

.real-label-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1a3a1a, #2a5a2a);
    border: 1px solid #4a9a4a;
    color: #7ecba1;
    font-size: 0.65rem;
    letter-spacing: 2px;
    text-transform: uppercase;
    padding: 3px 10px;
    border-radius: 2px;
    margin-left: 8px;
    vertical-align: middle;
}
</style>
""", unsafe_allow_html=True)

# ─── Matplotlib theme ──────────────────────────────────────────────────────────
OBSIDIAN   = '#080810'
CARD_BG    = '#0d0d1a'
GOLD       = '#d4af6b'
GOLD_LIGHT = '#f0d5a0'
GOLD_DEEP  = '#b8895a'
ROSE       = '#c9957a'
TEAL       = '#5ecfb0'
SAGE       = '#7ecba1'
MUTED      = '#3a3530'

GOLD_CMAP = LinearSegmentedColormap.from_list('gold', ['#1a1408','#6b4f1a','#d4af6b','#f0d5a0'])
ROSE_CMAP = LinearSegmentedColormap.from_list('rose', ['#120808','#6b2020','#c9957a','#f0c5b0'])
MULTI_COLORS = [GOLD, ROSE, TEAL, SAGE, '#a78bfa', '#f472b6', '#38bdf8', '#fb923c', '#a3e635', '#e879f9']

def style_fig(fig):
    fig.patch.set_facecolor(OBSIDIAN)
    return fig

def style_ax(ax, title='', xlabel='', ylabel='', grid=True):
    ax.set_facecolor(CARD_BG)
    ax.set_title(title, color=GOLD_LIGHT, fontsize=11,
                 fontfamily='serif', fontstyle='italic', pad=14)
    ax.set_xlabel(xlabel, color='#5a5248', fontsize=9, labelpad=8)
    ax.set_ylabel(ylabel, color='#5a5248', fontsize=9, labelpad=8)
    ax.tick_params(colors='#4a4540', labelsize=8)
    for sp in ax.spines.values():
        sp.set_edgecolor('#1e1c18')
    if grid:
        ax.grid(color='#1a1815', linewidth=.5, linestyle='--', alpha=.6)
    return ax

# ─── Dataset Setup (local or cloud download) ──────────────────────────────────
@st.cache_resource
def setup_dataset():
    folder = "myntradataset"

    # Try downloading from Google Drive if folder doesn't exist (cloud deployment)
    if not os.path.exists(folder):
        dataset_id = GDRIVE_FILES.get('dataset_zip', '')
        if dataset_id:
            download_from_gdrive(dataset_id, 'myntradataset.zip', is_zip=True)
        else:
            st.error("⚠️ Dataset folder 'myntradataset' not found.")
            st.info("For cloud deployment, set the `GDRIVE_DATASET_ID` secret in Streamlit Cloud settings.")
            st.stop()

    # Handle nested extraction: if zip created myntradataset/myntradataset/
    nested = os.path.join(folder, "myntradataset")
    if os.path.isdir(nested) and not os.path.exists(os.path.join(folder, "images")):
        import shutil
        for item in os.listdir(nested):
            src = os.path.join(nested, item)
            dst = os.path.join(folder, item)
            if not os.path.exists(dst):
                shutil.move(src, dst)
        shutil.rmtree(nested, ignore_errors=True)

    if not os.path.exists(f"{folder}/images"):
        st.error("'images' folder missing inside myntradataset.")
        st.stop()

    if not os.path.exists(f"{folder}/styles.csv"):
        st.warning("styles.csv not found (labels may be missing)")

    # Log image count for debugging
    img_dir = os.path.join(folder, "images")
    img_count = len([f for f in os.listdir(img_dir) if f.endswith(('.jpg','.jpeg','.png'))])
    st.sidebar.caption(f"📸 {img_count:,} images found on disk")

    return folder

# ─── Feature extractor ────────────────────────────────────────────────────────
class FeatureExtractor:
    def __init__(self, name='resnet18'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        base = (models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) if name=='resnet50'
                else models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1))
        self.model = nn.Sequential(*list(base.children())[:-1]).to(self.device).eval()
        self.tf = transforms.Compose([
            transforms.Resize((224,224)), transforms.ToTensor(),
            transforms.Normalize([.485,.456,.406],[.229,.224,.225])
        ])
    def extract_features(self, img):
        if isinstance(img, str):          img = Image.open(img)
        elif isinstance(img, np.ndarray): img = Image.fromarray(img)
        if img.mode != 'RGB': img = img.convert('RGB')
        t = self.tf(img).unsqueeze(0).to(self.device)
        with torch.no_grad(): return self.model(t).squeeze().cpu().numpy()

# ─── Recommender ──────────────────────────────────────────────────────────────
class FashionRecommender:
    def __init__(self, features, metadata):
        self.features = features; self.metadata = metadata
        # L2-normalize for memory-efficient cosine via euclidean on unit vectors
        norms = np.linalg.norm(features, axis=1, keepdims=True)
        norms[norms == 0] = 1
        self.features = (features / norms).astype(np.float32)
        self.knn = NearestNeighbors(n_neighbors=min(20, len(features)),
                                    metric='euclidean', algorithm='ball_tree')
        self.knn.fit(self.features)

    def get_recommendations(self, idx, n=6):
        d, i = self.knn.kneighbors(self.features[idx].reshape(1,-1), n_neighbors=n+1)
        rec = self.metadata.iloc[i[0][1:]].copy()
        rec['similarity_score'] = np.clip(1 - (d[0][1:] ** 2) / 2, 0, 1)
        return rec

    def find_similar_to_uploaded(self, f, n=6):
        norm = np.linalg.norm(f)
        f_norm = (f / norm if norm > 0 else f).astype(np.float32).reshape(1, -1)
        d, i = self.knn.kneighbors(f_norm, n_neighbors=n)
        rec = self.metadata.iloc[i[0]].copy()
        rec['similarity_score'] = np.clip(1 - (d[0] ** 2) / 2, 0, 1)
        return rec

# ─── Model config ─────────────────────────────────────────────────────────────
MODEL_OPTIONS = {
    'ResNet-50 (Recommended)': {
        'pkl':        'models/fashion_recommender_resnet50.pkl',
        'extractor':  'resnet50',
        'feat_dim':   '2048-D',
        'gdrive_key': 'resnet50_pkl',
        'desc':       'Best balance of accuracy & speed',
        'size':       '~354 MB',
    },
    'ResNet-18 (Lightweight)': {
        'pkl':        'models/fashion_recommender_resnet18.pkl',
        'extractor':  'resnet18',
        'feat_dim':   '512-D',
        'gdrive_key': 'resnet18_pkl',
        'desc':       'Fastest inference, lower memory',
        'size':       '~93 MB',
    },
    'VGG-16 (High Fidelity)': {
        'pkl':        'models/fashion_recommender_vgg16.pkl',
        'extractor':  'vgg16',
        'feat_dim':   '4096-D',
        'gdrive_key': 'vgg16_pkl',
        'desc':       'Highest feature depth, most memory',
        'size':       '~701 MB',
    },
}

# ─── Load model + merge styles.csv ────────────────────────────────────────────
# NOTE: Not cached with @st.cache_resource to allow model switching without OOM
#       on Streamlit Cloud free tier (1GB RAM). Only one model stays in memory.
def load_model(model_name: str = 'ResNet-50 (Recommended)'):
    import gc
    gc.collect()  # Free memory from previous model

    cfg      = MODEL_OPTIONS[model_name]
    pkl_path = cfg['pkl']

    # Try downloading from Google Drive if model file doesn't exist
    if not os.path.exists(pkl_path):
        os.makedirs('models', exist_ok=True)
        gdrive_key = cfg.get('gdrive_key', '')
        file_id = GDRIVE_FILES.get(gdrive_key, '')
        if file_id:
            with st.spinner(f'Downloading {model_name}... (first time only)'):
                download_from_gdrive(file_id, pkl_path)
        if not os.path.exists(pkl_path):
            st.error(f"Model file not found: {pkl_path}\nPlease set the correct Google Drive secret.")
            st.stop()

    try:
        with st.spinner(f'Loading {model_name}...'):
            data     = joblib.load(pkl_path)
            features = data['features']
            metadata = data['metadata']
            del data  # Free the raw dict
            gc.collect()

            if features.dtype != np.float32:
                features = features.astype(np.float32)
            if 'image_path' in metadata.columns:
                # Normalize paths: strip Colab prefix (/content/my_dataset/) and Windows backslashes
                metadata['image_path'] = (
                    metadata['image_path']
                    .str.replace('\\', '/', regex=False)
                    .str.replace(r'^.*/myntradataset/', 'myntradataset/', regex=True)
                )

            # ── Merge styles.csv for real labels (only if not already in pkl) ─────
            if 'productDisplayName' not in metadata.columns:
                styles_path = 'myntradataset/styles.csv'
                if os.path.exists(styles_path):
                    styles = pd.read_csv(styles_path, on_bad_lines='skip')
                    styles['id'] = styles['id'].astype(str)
                    metadata['image_id'] = metadata['image_id'].astype(str)
                    metadata = metadata.merge(styles, left_on='image_id', right_on='id', how='left')

            extractor_type = cfg['extractor']
            return FashionRecommender(features, metadata), FeatureExtractor(extractor_type), metadata, features
    except Exception as e:
        st.error(f"Failed to load {model_name}: {str(e)}")
        st.info("This model may be too large for the free tier, or the file may be corrupted. Try a different model.")
        st.stop()

# ═════════════════════════════════════════════════════════════════════════════
#   BROWSE MODE
# ═════════════════════════════════════════════════════════════════════════════
def browse_catalog_mode(rec, meta):
    st.markdown('<div class="section-title">Browse the Catalog</div><div class="section-rule"></div>', unsafe_allow_html=True)
    c1,c2 = st.columns([3,1])
    with c1: st.markdown(f'<p style="color:#6a6058;font-size:.8rem;letter-spacing:2px;text-transform:uppercase">{len(meta):,} items indexed</p>', unsafe_allow_html=True)
    with c2: n = st.slider("Results", 3, 9, 6)
    if 'sample' not in st.session_state or st.button("↺  Refresh"):
        st.session_state.sample = meta.sample(min(10, len(meta)))
    cols = st.columns(5); sel = None
    for i,(_, item) in enumerate(st.session_state.sample.iterrows()):
        with cols[i%5]:
            try:
                st.image(Image.open(item['image_path']), width='stretch')
                name = item.get('productDisplayName', item['filename'])
                st.caption(str(name)[:30] if pd.notna(name) else item['filename'][:25])
                if st.button("Select", key=f"s{i}"): sel = item.name
            except: pass
    if sel is not None:
        st.markdown('<hr style="border-color:#1e1c18;margin:2rem 0">', unsafe_allow_html=True)
        q = meta.iloc[sel]
        c1,c2 = st.columns([1,2])
        with c1:
            st.markdown('<p style="font-family:serif;font-style:italic;color:#d4af6b;font-size:1rem">Query Item</p>', unsafe_allow_html=True)
            try: st.image(Image.open(q['image_path']), width='stretch')
            except: pass
            if 'productDisplayName' in q and pd.notna(q['productDisplayName']):
                st.markdown(f'<p style="font-size:.8rem;color:#a89880">{q["productDisplayName"]}</p>', unsafe_allow_html=True)
            for label in ['masterCategory','subCategory','articleType','baseColour','season']:
                if label in q and pd.notna(q[label]):
                    st.markdown(f'<span style="font-size:.7rem;color:#6a6058;text-transform:uppercase;letter-spacing:1px">{label}:</span> <span style="font-size:.8rem;color:#d4af6b">{q[label]}</span><br>', unsafe_allow_html=True)
        with c2:
            st.markdown('<p style="font-family:serif;font-style:italic;color:#d4af6b;font-size:1rem">Similar Items</p>', unsafe_allow_html=True)
            recs = rec.get_recommendations(sel, n)
            rc = st.columns(3)
            for i,(_, r) in enumerate(recs.iterrows()):
                with rc[i%3]:
                    try:
                        st.image(Image.open(r['image_path']), width='stretch')
                        score = r['similarity_score']
                        color = '#7ecba1' if score>.85 else ('#d4af6b' if score>.7 else '#e07070')
                        st.markdown(f'<p style="color:{color};font-size:.8rem;font-weight:600;text-align:center;margin:0">{score:.1%}</p>', unsafe_allow_html=True)
                        if 'productDisplayName' in r and pd.notna(r['productDisplayName']):
                            st.caption(str(r['productDisplayName'])[:28])
                    except: pass

# ═════════════════════════════════════════════════════════════════════════════
#   UPLOAD MODE
# ═════════════════════════════════════════════════════════════════════════════
def upload_image_mode(rec, ext, meta):
    st.markdown('<div class="section-title">Upload & Discover</div><div class="section-rule"></div>', unsafe_allow_html=True)
    f = st.file_uploader("Drop your image here", type=['jpg','jpeg','png'])
    if f:
        img = Image.open(f)
        if img.mode!='RGB': img=img.convert('RGB')
        c1,c2 = st.columns([1,2])
        with c1:
            st.markdown('<p style="font-family:serif;font-style:italic;color:#d4af6b">Your Image</p>', unsafe_allow_html=True)
            st.image(img, width='stretch')
        with c2:
            st.markdown('<p style="font-family:serif;font-style:italic;color:#d4af6b">Closest Matches</p>', unsafe_allow_html=True)
            n = st.slider("Results", 3, 12, 6)
            with st.spinner("Extracting features…"):
                feats = ext.extract_features(img)
            recs = rec.find_similar_to_uploaded(feats, n)
            rc = st.columns(3)
            for i,(_, r) in enumerate(recs.iterrows()):
                with rc[i%3]:
                    try:
                        st.image(Image.open(r['image_path']), width='stretch')
                        st.markdown(f'<p style="color:#d4af6b;font-size:.8rem;text-align:center;margin:0">{r["similarity_score"]:.1%}</p>', unsafe_allow_html=True)
                        if 'productDisplayName' in r and pd.notna(r.get('productDisplayName',None)):
                            st.caption(str(r['productDisplayName'])[:28])
                    except: pass

# ═════════════════════════════════════════════════════════════════════════════
#   ANALYTICS DASHBOARD
# ═════════════════════════════════════════════════════════════════════════════
def analytics_dashboard(meta, rec, features):

    has_labels = 'masterCategory' in meta.columns and meta['masterCategory'].notna().sum() > 100

    st.markdown('<div class="section-title">Analytics & Evaluation</div><div class="section-rule"></div>', unsafe_allow_html=True)
    if has_labels:
        st.markdown('<span class="real-label-badge">✦ Real Labels Active — styles.csv merged</span>', unsafe_allow_html=True)

    tab1, = st.tabs([
        "  ✦  Dataset & EDA  ",
    ])

    # ═══════ TAB 1 — EDA ══════════════════════════════════════════════════════
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)

        # KPI row
        cats   = meta['masterCategory'].nunique() if has_labels else "—"
        colors = meta['baseColour'].nunique()      if has_labels and 'baseColour' in meta.columns else "—"
        types  = meta['articleType'].nunique()     if has_labels and 'articleType' in meta.columns else "—"

        kpis = [
            ("Total Items",      f"{len(meta):,}",    "Fashion images"),
            ("Categories",       str(cats),            "Master categories"),
            ("Article Types",    str(types),           "Unique article types"),
            ("Colour Variants",  str(colors),          "Base colours"),
        ]
        cols = st.columns(4)
        for col,(label,val,sub) in zip(cols,kpis):
            with col:
                st.markdown(f"""<div class="kpi-card">
                    <span class="kpi-number">{val}</span>
                    <div class="kpi-label">{label}</div>
                    <div class="kpi-sub">{sub}</div>
                </div>""", unsafe_allow_html=True)

        if has_labels:
            st.markdown("---")

            # ── Master Category distribution ─────────────────────────────────
            st.markdown('<div class="section-title" style="font-size:1.3rem">Master Category Distribution</div><div class="section-rule"></div>', unsafe_allow_html=True)
            cat_counts = meta['masterCategory'].value_counts()

            fig, axes = plt.subplots(1, 2, figsize=(15, 5)); style_fig(fig)
            # Bar chart
            bar_colors = [MULTI_COLORS[i % len(MULTI_COLORS)] for i in range(len(cat_counts))]
            axes[0].barh(cat_counts.index, cat_counts.values,
                         color=bar_colors, edgecolor=OBSIDIAN, lw=.5, height=.7)
            for i, (val, lbl) in enumerate(zip(cat_counts.values, cat_counts.index)):
                axes[0].text(val + len(meta)*0.002, i, f'{val:,}',
                             va='center', color=GOLD_LIGHT, fontsize=8)
            style_ax(axes[0], 'Items per Master Category', 'Count', '', grid=False)
            axes[0].invert_yaxis()

            # Donut
            wedge_colors = [MULTI_COLORS[i % len(MULTI_COLORS)] for i in range(len(cat_counts))]
            wedges, texts, autotexts = axes[1].pie(
                cat_counts.values, labels=cat_counts.index,
                colors=wedge_colors, autopct='%1.1f%%',
                pctdistance=.82, startangle=90,
                wedgeprops=dict(width=.55, edgecolor=OBSIDIAN, linewidth=1.5)
            )
            for t in texts:    t.set_color(GOLD_LIGHT); t.set_fontsize(8)
            for at in autotexts: at.set_color(OBSIDIAN); at.set_fontsize(7); at.set_fontweight('bold')
            axes[1].set_facecolor(CARD_BG)
            axes[1].set_title('Category Share', color=GOLD_LIGHT, fontsize=11,
                              fontfamily='serif', fontstyle='italic')

            # Use legend instead of inline labels to avoid overlap
            axes[1].legend(wedges, [f'{l} ({v:,})' for l, v in zip(cat_counts.index, cat_counts.values)],
                           loc='center left', bbox_to_anchor=(1.05, 0.5),
                           facecolor=CARD_BG, labelcolor=GOLD_LIGHT, fontsize=7,
                           edgecolor=MUTED, framealpha=0.9)
            # Hide the inline labels that overlap
            for t in texts: t.set_text('')

            fig.tight_layout(pad=2); st.pyplot(fig, width='stretch'); plt.close()

            st.markdown("---")

            # ── Article Type top-15 ──────────────────────────────────────────
            st.markdown('<div class="section-title" style="font-size:1.3rem">Top 15 Article Types</div><div class="section-rule"></div>', unsafe_allow_html=True)
            art_counts = meta['articleType'].value_counts().head(15)
            fig, ax = plt.subplots(figsize=(13, 5)); style_fig(fig)
            bars = ax.bar(art_counts.index, art_counts.values,
                          color=[GOLD_CMAP(i/15) for i in range(15)],
                          edgecolor=OBSIDIAN, lw=.5, width=.7)
            for bar,v in zip(bars, art_counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+30, str(v),
                        ha='center', va='bottom', color=GOLD_LIGHT, fontsize=7.5)
            ax.set_xticklabels(art_counts.index, rotation=35, ha='right', fontsize=8)
            style_ax(ax, 'Top 15 Article Types by Count', '', 'Count')
            fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close()

            st.markdown("---")

            # ── Gender + Season + Usage ──────────────────────────────────────
            st.markdown('<div class="section-title" style="font-size:1.3rem">Gender · Season · Usage</div><div class="section-rule"></div>', unsafe_allow_html=True)

            chart_cols = st.columns(3)
            for ci, (col_, title) in enumerate(zip(
                ['gender','season','usage'],
                ['Gender Distribution','Season Distribution','Usage Distribution'])):
                if col_ not in meta.columns: continue
                vc = meta[col_].value_counts().dropna()

                # Group tiny slices (<2%) into "Other"
                threshold = vc.sum() * 0.02
                main = vc[vc >= threshold]
                other_sum = vc[vc < threshold].sum()
                if other_sum > 0:
                    main = pd.concat([main, pd.Series({'Other': other_sum})])

                wedge_c = [MULTI_COLORS[i % len(MULTI_COLORS)] for i in range(len(main))]
                fig, ax = plt.subplots(figsize=(6, 5)); style_fig(fig)
                wedges, texts, autotexts = ax.pie(
                    main.values, labels=None, colors=wedge_c,
                    autopct='%1.1f%%', pctdistance=.82, startangle=90,
                    wedgeprops=dict(width=.5, edgecolor=OBSIDIAN, linewidth=1.2)
                )
                for at in autotexts: at.set_color(GOLD_LIGHT); at.set_fontsize(8); at.set_fontweight('bold')
                ax.set_facecolor(CARD_BG)
                ax.set_title(title, color=GOLD_LIGHT, fontsize=11,
                             fontfamily='serif', fontstyle='italic', pad=12)
                ax.legend(wedges, [f'{l}  ({v:,})' for l, v in zip(main.index, main.values)],
                          loc='upper center', bbox_to_anchor=(0.5, -0.02), ncol=2,
                          facecolor=CARD_BG, labelcolor=GOLD_LIGHT, fontsize=8,
                          edgecolor=MUTED, framealpha=0.9)
                fig.tight_layout(pad=1.5)
                with chart_cols[ci]:
                    st.pyplot(fig, width='stretch')
                plt.close()

            st.markdown("---")

            # ── Base Colour top-20 ───────────────────────────────────────────
            st.markdown('<div class="section-title" style="font-size:1.3rem">Colour Distribution (Top 20)</div><div class="section-rule"></div>', unsafe_allow_html=True)
            colour_counts = meta['baseColour'].value_counts().head(20)

            # Map colour names to hex roughly
            colour_map = {
                'Black':'#1a1a1a','White':'#f5f5f0','Navy Blue':'#1a2a5e',
                'Blue':'#3b6bce','Grey':'#808080','Red':'#cc2222',
                'Green':'#2d7a2d','Beige':'#d4b896','Brown':'#7a4a1e',
                'Pink':'#e87fa0','Yellow':'#e8cc22','Orange':'#e87722',
                'Purple':'#6a2aaa','Silver':'#c0c0c0','Gold':'#d4af6b',
                'Maroon':'#800000','Olive':'#808000','Teal':'#008080',
                'Cream':'#fffdd0','Khaki':'#c3b091'
            }
            bar_c = [colour_map.get(c, GOLD_DEEP) for c in colour_counts.index]
            fig, ax = plt.subplots(figsize=(13, 5)); style_fig(fig)
            bars = ax.bar(colour_counts.index, colour_counts.values,
                          color=bar_c, edgecolor='#1a1a1a', lw=.8, width=.7)
            for bar,v in zip(bars, colour_counts.values):
                ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+20, str(v),
                        ha='center', va='bottom', color=GOLD_LIGHT, fontsize=7.5)
            ax.set_xticklabels(colour_counts.index, rotation=35, ha='right', fontsize=8)
            style_ax(ax, 'Top 20 Colours in Dataset', '', 'Count')
            fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close()

            st.markdown("---")

            # ── Year trend ───────────────────────────────────────────────────
            if 'year' in meta.columns:
                st.markdown('<div class="section-title" style="font-size:1.3rem">Items Added by Year</div><div class="section-rule"></div>', unsafe_allow_html=True)
                year_counts = meta['year'].dropna().astype(int).value_counts().sort_index()
                fig, ax = plt.subplots(figsize=(12, 4)); style_fig(fig)
                ax.plot(year_counts.index, year_counts.values, color=GOLD, lw=2.5, zorder=3, marker='o', markersize=5)
                ax.fill_between(year_counts.index, year_counts.values, alpha=.12, color=GOLD)
                style_ax(ax, 'Dataset Growth by Year', 'Year', 'Number of Items')
                fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close()

        # ── PCA ──────────────────────────────────────────────────────────────
        st.markdown("---")
        st.markdown('<div class="section-title" style="font-size:1.3rem">Feature Space — PCA Projection</div><div class="section-rule"></div>', unsafe_allow_html=True)
        with st.spinner("Running PCA…"):
            ps  = min(3000, len(features))
            pi  = np.random.choice(len(features), ps, replace=False)
            pca = PCA(n_components=2, random_state=42)
            coords = pca.fit_transform(features[pi])

        fig, ax = plt.subplots(figsize=(11,6)); style_fig(fig)
        if has_labels and 'masterCategory' in meta.columns:
            cats_list = meta['masterCategory'].iloc[pi].fillna('Unknown').values
            unique_cats = list(pd.Series(cats_list).unique())
            c_map = {c: MULTI_COLORS[i % len(MULTI_COLORS)] for i,c in enumerate(unique_cats)}
            point_colors = [c_map[c] for c in cats_list]
            for cat in unique_cats:
                mask = [c==cat for c in cats_list]
                ax.scatter(coords[mask,0], coords[mask,1], c=c_map[cat],
                           alpha=.4, s=5, linewidths=0, label=cat, zorder=2)
            ax.legend(facecolor=CARD_BG, labelcolor=GOLD_LIGHT, fontsize=7,
                      edgecolor=MUTED, markerscale=2, ncol=2)
        else:
            magnitudes = np.linalg.norm(features[pi], axis=1)
            sc = ax.scatter(coords[:,0], coords[:,1], c=magnitudes,
                            cmap=GOLD_CMAP, alpha=.35, s=4, linewidths=0, zorder=2)
            plt.colorbar(sc, ax=ax)
        var = pca.explained_variance_ratio_
        style_ax(ax, f'ResNet18 Embeddings — PC1 {var[0]:.1%}  ·  PC2 {var[1]:.1%}  ·  {ps:,} items',
                 'Principal Component 1', 'Principal Component 2')
        fig.tight_layout(); st.pyplot(fig, width='stretch'); plt.close()

        st.markdown(f"""<div class="insight">
        Each color represents a <b>master category</b>. Visible clustering confirms the model
        learned visually coherent style representations — items of the same category group together
        in the 512-D embedding space.
        </div>""", unsafe_allow_html=True)
# ═════════════════════════════════════════════════════════════════════════════
#   MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    st.markdown("""
    <div style="padding:2.5rem 0 1rem 0;">
        <div class="hero-title">Fashion Intelligence</div>
        <div class="hero-sub">Visual Recommendation System · Myntra Dataset</div>
        <div class="hero-divider"></div>
    </div>
    """, unsafe_allow_html=True)

    setup_dataset()

    # ── Navigation (sidebar) ────────────────────────────────────────────────────
    st.sidebar.markdown("""
    <div style="font-family:'Cormorant Garamond',serif;font-style:italic;
                font-size:1.4rem;color:#d4af6b;margin-bottom:1rem;">Navigation</div>
    """, unsafe_allow_html=True)
    mode = st.sidebar.radio("Navigation", ["Browse Catalog","Upload Image","Analytics"],
                            label_visibility="collapsed")
    st.sidebar.markdown('<hr style="border-color:#1e1c18;margin:1.5rem 0">', unsafe_allow_html=True)

    # ── Model selector dropdown (default = ResNet-50) ─────────────────────────
    st.sidebar.markdown("""
    <div style="font-size:.65rem;color:#6a6058;letter-spacing:2px;text-transform:uppercase;
                margin-bottom:.4rem;">⚙ Select Model</div>
    """, unsafe_allow_html=True)
    model_names = list(MODEL_OPTIONS.keys())
    model_choice = st.sidebar.selectbox(
        "Choose a backbone model",
        model_names,
        index=0,   # ResNet-50 (Recommended) is first / default
        help="Select the deep learning model for feature extraction. ResNet-50 offers the best accuracy-speed trade-off.",
    )

    # Show selected model info card
    cfg = MODEL_OPTIONS[model_choice]
    st.sidebar.markdown(f"""
    <div style="background:rgba(212,175,107,0.06); border:1px solid rgba(212,175,107,0.15);
                border-radius:4px; padding:0.8rem 1rem; margin:0.6rem 0 0.3rem 0;">
        <div style="font-size:.72rem;color:#d4af6b;font-weight:500;margin-bottom:4px;">
            {model_choice.split(' (')[0]}</div>
        <div style="font-size:.68rem;color:#8a7f72;line-height:1.7;">
            {cfg['desc']}<br>
            Features: <span style="color:#d4af6b">{cfg['feat_dim']}</span> · 
            Size: <span style="color:#d4af6b">{cfg['size']}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    recommender, extractor, metadata, features = load_model(model_choice)

    st.sidebar.markdown('<hr style="border-color:#1e1c18;margin:1.5rem 0">', unsafe_allow_html=True)
    has_labels = 'masterCategory' in metadata.columns
    backbone_name = model_choice.split(' (')[0]
    st.sidebar.markdown(f"""
    <div style="font-size:.7rem;color:#3a3530;letter-spacing:1px;line-height:2.2">
    DATASET &nbsp;·&nbsp; Myntra<br>
    BACKBONE &nbsp;·&nbsp; {backbone_name}<br>
    ITEMS &nbsp;·&nbsp; {len(metadata):,}<br>
    FEATURES &nbsp;·&nbsp; {cfg['feat_dim']}<br>
    LABELS &nbsp;·&nbsp; {'Real ✦' if has_labels else 'KMeans'}
    </div>""", unsafe_allow_html=True)

    if mode == "Browse Catalog":
        browse_catalog_mode(recommender, metadata)
    elif mode == "Upload Image":
        upload_image_mode(recommender, extractor, metadata)
    else:
        analytics_dashboard(metadata, recommender, features)

if __name__ == "__main__":
    main()