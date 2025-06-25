import streamlit as st
from PIL import Image
import os

from src.recommendation.engine import RecommendationEngine
from src.vector_database.vectorstore import VectorStore
from src.data.get_data import getData

# -------------- Config --------------
st.set_page_config(page_title="🧥 Men's Fashion Recommender", layout="wide")
IMAGES_PER_PAGE = 12

# -------------- Ensure Dataset Exists (once) --------------
@st.cache_resource
def initialize_data():
    getData()
    return VectorStore(), RecommendationEngine()

vector_store, recommendation_engine = initialize_data()

# -------------- Session State Defaults --------------
session_defaults = {
    "liked": {},
    "disliked": {},
    "current_page": 0,
    "recommended_images": vector_store.points,
    "vector_store": vector_store,
    "recommendation_engine": recommendation_engine,
}

for key, value in session_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = value

# -------------- Sidebar Info --------------
with st.sidebar:
    st.title("🧥 Men's Fashion Recommender")

    st.markdown("""
    **Discover fashion styles that suit your taste.**  
    Like 👍 or dislike 👎 outfits and receive AI-powered recommendations tailored to you.
    """)

    st.markdown("### 📦 Dataset")
    st.markdown("""
    - Source: [Kaggle – virat164/fashion-database](https://www.kaggle.com/datasets/virat164/fashion-database)  
    - ~2,000 fashion images
    """)

    st.markdown("### 🧠 How It Works")
    st.markdown("""
    1. Images are embedded into vector space  
    2. You provide preferences via Like/Dislike  
    3. Qdrant finds visually similar images  
    4. Results are updated in real-time
    """)

    st.markdown("### ⚙️ Technologies")
    st.markdown("""
    - **Streamlit** UI  
    - **Qdrant** vector DB  
    - **Python** backend  
    - **PIL** for image handling  
    - **Kaggle API** for data
    """)

    st.markdown("---")
    st.markdown("🔗 [Connect on LinkedIn](https://www.linkedin.com/in/rindra-ny-aiko-randriamihamina/)")

# -------------- Core Logic Functions --------------
def get_recommendations(liked_ids, disliked_ids):
    return st.session_state.recommendation_engine.get_recommendations(
        liked_images=liked_ids,
        disliked_images=disliked_ids,
        limit=3 * IMAGES_PER_PAGE
    )

def refresh_recommendations():
    liked_ids = list(st.session_state.liked.keys())
    disliked_ids = list(st.session_state.disliked.keys())
    st.session_state.recommended_images = get_recommendations(liked_ids, disliked_ids)

# -------------- Display: Selected Preferences --------------
def display_selected_images():
    if not st.session_state.liked and not st.session_state.disliked:
        return

    st.markdown("### 🧍 Your Picks")
    cols = st.columns(6)
    images = st.session_state.vector_store.points

    for i, (img_id, status) in enumerate(
        list(st.session_state.liked.items()) + list(st.session_state.disliked.items())
    ):
        img_path = next((img["image_path"] for img in images if img["id"] == img_id), None)
        if img_path and os.path.exists(img_path):
            with cols[i % 6]:
                st.image(img_path, use_container_width=True, caption=f"{img_id} ({status})")
                col1, col2 = st.columns(2)
                if col1.button("❌ Remove", key=f"remove_{img_id}"):
                    if status == "liked":
                        del st.session_state.liked[img_id]
                    else:
                        del st.session_state.disliked[img_id]
                    refresh_recommendations()
                    st.rerun()

                if col2.button("🔁 Switch", key=f"switch_{img_id}"):
                    if status == "liked":
                        del st.session_state.liked[img_id]
                        st.session_state.disliked[img_id] = "disliked"
                    else:
                        del st.session_state.disliked[img_id]
                        st.session_state.liked[img_id] = "liked"
                    refresh_recommendations()
                    st.rerun()

# -------------- Display: Recommended Gallery --------------
def display_gallery():
    st.markdown("### 🧠 Smart Suggestions")

    page = st.session_state.current_page
    start_idx = page * IMAGES_PER_PAGE
    end_idx = start_idx + IMAGES_PER_PAGE
    current_images = st.session_state.recommended_images[start_idx:end_idx]

    cols = st.columns(4)
    for idx, img in enumerate(current_images):
        with cols[idx % 4]:
            if os.path.exists(img["image_path"]):
                st.image(img["image_path"], use_container_width=True)
            else:
                st.warning("Image not found")

            col1, col2 = st.columns(2)
            if col1.button("👍 Like", key=f"like_{img['id']}"):
                st.session_state.liked[img["id"]] = "liked"
                refresh_recommendations()
                st.rerun()
            if col2.button("👎 Dislike", key=f"dislike_{img['id']}"):
                st.session_state.disliked[img["id"]] = "disliked"
                refresh_recommendations()
                st.rerun()

    # Pagination
    col1, _, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous") and page > 0:
            st.session_state.current_page -= 1
            st.rerun()
    with col3:
        if st.button("➡️ Next") and end_idx < len(st.session_state.recommended_images):
            st.session_state.current_page += 1
            st.rerun()

# -------------- Main Render Pipeline --------------
st.title("🧥 Men's Fashion Recommender")

display_selected_images()
st.divider()
display_gallery()
