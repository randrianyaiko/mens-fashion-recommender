import streamlit as st
from PIL import Image
import os
from src.recommendation.engine import RecommendationEngine
from src.vector_database.vectorstore import VectorStore
from src.data.get_data import getData

# Ensure the dataset is available
getData()

# Initialize components
recommendation_engine = RecommendationEngine()
vector_store = VectorStore()
images = vector_store.points  # This should be a list of dicts with "id" and "image_path"

# -------------- Config --------------
st.set_page_config(page_title="🧥 Men's Fashion Recommender", layout="wide")
IMAGES_PER_PAGE = 12

# -------------- Session State Initialization --------------
if "liked" not in st.session_state:
    st.session_state.liked = {}

if "disliked" not in st.session_state:
    st.session_state.disliked = {}

if "current_page" not in st.session_state:
    st.session_state.current_page = 0

if "recommended_images" not in st.session_state:
    st.session_state.recommended_images = images  # Initial fallback

# -------------- Recommendation Logic Integration --------------
def get_recommendations(liked_ids, disliked_ids):
    return recommendation_engine.get_recommendations(
        liked_images=liked_ids,
        disliked_images=disliked_ids,
        limit=3*IMAGES_PER_PAGE
    )

# -------------- Refresh Recommendations --------------
def refresh_recommendations():
    liked_ids = list(st.session_state.liked.keys())
    disliked_ids = list(st.session_state.disliked.keys())
    st.session_state.recommended_images = get_recommendations(liked_ids, disliked_ids)

# -------------- Top: Liked / Disliked Images --------------
def display_selected_images():
    if not st.session_state.liked and not st.session_state.disliked:
        return

    st.markdown("### 🎯 🧍 Your Picks")
    cols = st.columns(6)
    i = 0
    for img_id, status in list(st.session_state.liked.items()) + list(st.session_state.disliked.items()):
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
        i += 1

# -------------- Gallery Section --------------
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

    # Pagination controls
    col1, col2, col3 = st.columns([1, 2, 1])
    with col1:
        if st.button("⬅️ Previous") and page > 0:
            st.session_state.current_page -= 1
            st.rerun()
    with col3:
        if st.button("➡️ Next") and end_idx < len(st.session_state.recommended_images):
            st.session_state.current_page += 1
            st.rerun()

# -------------- Render App --------------
st.title("🧥 Men's Fashion Recommender")

display_selected_images()
st.divider()
display_gallery()