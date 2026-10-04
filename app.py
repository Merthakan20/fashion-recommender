"""Streamlit web arayüzü.

    streamlit run app.py
"""

import os
import tempfile
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow log kalabalığını gizle

import streamlit as st  # noqa: E402

from fashion_rec.config import IMAGE_EXTENSIONS  # noqa: E402
from fashion_rec.features import build_index, load_model  # noqa: E402
from fashion_rec.recommend import recommend  # noqa: E402
from fashion_rec.text_search import build_text_index, load_clip_model, search  # noqa: E402

GRID_COLUMNS = 4


@st.cache_resource(show_spinner="Görsel özellikleri hazırlanıyor...")
def get_index():
    return build_index()


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Metin arama modeli (CLIP) yükleniyor...")
def get_clip_model():
    return load_clip_model()


@st.cache_resource(show_spinner="Metin arama indeksi hazırlanıyor...")
def get_text_index():
    return build_text_index(model=get_clip_model())


def label(path: Path) -> str:
    """Dosya adından okunabilir etiket: uzantısız, ilk harfi büyük."""
    return path.stem[:1].upper() + path.stem[1:]


def recommend_for_upload(uploaded, paths, features, top_n):
    """Yüklenen dosyayı geçici olarak diske yazıp öneri hesaplar."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
    try:
        return recommend(tmp.name, paths, features, top_n=top_n, model=get_model())
    finally:
        os.unlink(tmp.name)


def show_grid(results):
    for start in range(0, len(results), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (rank, (path, score)) in zip(cols, enumerate(results[start:start + GRID_COLUMNS], start=start + 1)):
            with col:
                st.image(str(path), width="stretch")
                st.markdown(f"**{rank}.** {label(path)}")
                st.progress(max(0.0, min(score, 1.0)), text=f"Benzerlik: {score:.3f}")


st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")
st.title("👗 Fashion Recommender")
st.caption("Bir kıyafet seçin, yükleyin veya tarif edin; en çok benzeyen kıyafetleri önerelim.")

paths, features = get_index()

with st.sidebar:
    st.header("Ayarlar")
    source = st.radio("Arama şekli", ["Veri setinden seç", "Görsel yükle", "Metinle ara"])
    top_n = st.slider("Öneri sayısı", min_value=1, max_value=12, value=GRID_COLUMNS)
    st.caption(f"Veri setinde {len(paths)} görsel var.")

if source == "Metinle ara":
    query = st.text_input("Aradığınız kıyafeti tarif edin (İngilizce)", placeholder="red sequined dress").strip()
    if not query:
        st.info("Başlamak için bir tarif yazın, ör. *black lace dress* veya *white trousers*.")
    else:
        text_paths, text_features = get_text_index()
        st.subheader("Sonuçlar")
        show_grid(search(query, text_paths, text_features, top_n=top_n, model=get_clip_model()))
else:
    query_image = None
    results = []

    if source == "Veri setinden seç":
        selected = st.selectbox("Kıyafet", paths, format_func=label)
        query_image = selected
        results = recommend(selected, paths, features, top_n=top_n)
    else:
        uploaded = st.file_uploader("Görsel yükle", type=[ext.lstrip(".") for ext in IMAGE_EXTENSIONS])
        if uploaded is None:
            st.info("Başlamak için bir kıyafet görseli yükleyin.")
        else:
            query_image = uploaded
            with st.spinner("Benzer kıyafetler aranıyor..."):
                results = recommend_for_upload(uploaded, paths, features, top_n)

    if query_image is not None:
        left, right = st.columns([1, 3], gap="large")
        with left:
            st.subheader("Seçilen")
            st.image(query_image if not isinstance(query_image, Path) else str(query_image), width="stretch")
        with right:
            st.subheader("Öneriler")
            show_grid(results)
