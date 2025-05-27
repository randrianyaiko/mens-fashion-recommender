# 🧥 Men's Fashion Recommender

An interactive web app that recommends men's fashion styles based on your personal preferences. Simply like 👍 or dislike 👎 outfits, and get smart, AI-powered fashion suggestions in real time.

![app-preview](https://storage.googleapis.com/s4a-prod-share-preview/default/st_app_screenshot_image/23f364fe-1a3d-483e-8faf-e98678dbc125/Raw_App_Screenshot.png?nf_resize=smartcrop&w=240&h=130) <!-- optional: add a screenshot -->

---

## 🚀 Live Demo

> The app is available here on [Streamlit Cloud](https://mens-fashion-recommender.streamlit.app/).

---

## 📦 Dataset

The app uses a curated dataset of fashion images from Kaggle:

- **Source**: [virat164/fashion-database](https://www.kaggle.com/datasets/virat164/fashion-database)
- Contains over 3,000+ high-quality fashion images
- Organized by fashion type, color, and item category

---

## 🧠 How It Works

1. **Feature Extraction**: Each fashion image is encoded into a vector using image embeddings.
2. **Vector Storage**: Vectors are stored in **Qdrant**, a high-performance vector search engine.
3. **Interactive Feedback**: Users like or dislike fashion images.
4. **Recommendation Engine**: Similar items are retrieved via vector similarity, updating in real time.

---

## ⚙️ Technologies Used

| Tool            | Purpose                          |
|-----------------|----------------------------------|
| `Streamlit`     | Frontend web app framework       |
| `Python`        | Core backend logic               |
| `PIL`           | Image loading and rendering      |
| `Qdrant`        | Vector similarity search engine  |
| `Kaggle API`    | Dataset access and download      |

---

## 📂 Project Structure

```bash
.
├── app.py                     # Main Streamlit app
├── src/
│   ├── data/
│   │   └── get_data.py        # Downloads and processes dataset
│   ├── recommendation/
│   │   └── engine.py          # Handles recommendation logic
│   └── vector_database/
│       └── vectorstore.py     # Handles vector storage/retrieval (Qdrant)
├── requirements.txt
└── README.md
