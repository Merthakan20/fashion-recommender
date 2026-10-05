"""Streamlit web arayüzü.

    streamlit run app.py
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow log kalabalığını gizle

import streamlit as st  # noqa: E402

from fashion_rec.config import IMAGE_EXTENSIONS  # noqa: E402
from fashion_rec.features import build_index, load_model  # noqa: E402
from fashion_rec.recommend import recommend  # noqa: E402
from fashion_rec.text_search import build_text_index, combined_search, load_clip_model, search  # noqa: E402

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


@contextmanager
def uploaded_file(uploaded):
    """Yüklenen dosyayı geçici olarak diske yazar, iş bitince siler."""
    suffix = Path(uploaded.name).suffix.lower()
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(uploaded.getvalue())
    try:
        yield Path(tmp.name)
    finally:
        os.unlink(tmp.name)


def find_similar(image_path, modifier, top_n, text_weight):
    """Değişiklik metni yoksa VGG16 ile, varsa CLIP ile görsel + metin birlikte arar."""
    if not modifier:
        # veri setindeki görselin vektörü önbellekte: model yalnızca yüklenen görsel için açılır
        model = None if image_path in paths else get_model()
        return recommend(image_path, paths, features, top_n=top_n, model=model)
    text_paths, text_features = get_text_index()
    return combined_search(image_path, modifier, text_paths, text_features, top_n=top_n,
                           text_weight=text_weight, model=get_clip_model())


def show_grid(results, combined=False):
    for start in range(0, len(results), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (rank, (path, score)) in zip(cols, enumerate(results[start:start + GRID_COLUMNS], start=start + 1)):
            with col:
                st.image(str(path), width="stretch")
                st.markdown(f"**{rank}.** {label(path)}")
                if combined:
                    # z-skorlarının ağırlıklı toplamı: 0–1 aralığında değil, yalnızca sıralama için
                    st.caption(f"Uyum skoru: {score:+.2f}")
                else:
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
    query = st.text_input("Aradığınız kıyafeti tarif edin (Türkçe veya İngilizce)", placeholder="kırmızı payetli elbise").strip()
    if not query:
        st.info("Başlamak için bir tarif yazın, ör. *siyah dantel elbise* veya *white trousers*.")
    else:
        text_paths, text_features = get_text_index()
        st.subheader("Sonuçlar")
        show_grid(search(query, text_paths, text_features, top_n=top_n, model=get_clip_model()))
else:
    if source == "Veri setinden seç":
        query_image = st.selectbox("Kıyafet", paths, format_func=label)
    else:
        query_image = st.file_uploader("Görsel yükle", type=[ext.lstrip(".") for ext in IMAGE_EXTENSIONS])

    left_in, right_in = st.columns([3, 1])
    with left_in:
        modifier = st.text_input("Değişiklik (isteğe bağlı): bunun gibi ama…",
                                 placeholder="kırmızı, uzun kollu, payetli…").strip()
    with right_in:
        text_weight = st.slider("Metin ağırlığı", 0.0, 1.0, 0.5, 0.1, disabled=not modifier,
                                help="0 = yalnız görsel, 1 = yalnız metin")

    if query_image is None:
        st.info("Başlamak için bir kıyafet görseli yükleyin.")
    else:
        with st.spinner("Benzer kıyafetler aranıyor..."):
            if isinstance(query_image, Path):
                results = find_similar(query_image, modifier, top_n, text_weight)
            else:
                with uploaded_file(query_image) as tmp_path:
                    results = find_similar(tmp_path, modifier, top_n, text_weight)

        left, right = st.columns([1, 3], gap="large")
        with left:
            st.subheader("Seçilen")
            st.image(query_image if not isinstance(query_image, Path) else str(query_image), width="stretch")
        with right:
            st.subheader(f"Öneriler: bunun gibi ama “{modifier}”" if modifier else "Öneriler")
            show_grid(results, combined=bool(modifier))
