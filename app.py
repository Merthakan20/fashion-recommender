"""Streamlit web arayüzü.

    streamlit run app.py
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow log kalabalığını gizle

import streamlit as st  # noqa: E402

from fashion_rec.color import build_color_index  # noqa: E402
from fashion_rec.config import IMAGE_EXTENSIONS  # noqa: E402
from fashion_rec.features import build_index, load_model  # noqa: E402
from fashion_rec.recommend import recommend  # noqa: E402
from fashion_rec.tags import TAG_GROUPS, build_tag_index, filter_results, tags_by_name  # noqa: E402
from fashion_rec.text_search import build_text_index, combined_search, load_clip_model, search  # noqa: E402

GRID_COLUMNS = 4


@st.cache_resource(show_spinner="Görsel özellikleri hazırlanıyor...")
def get_index():
    return build_index()


@st.cache_resource(show_spinner="Renk vektörleri hazırlanıyor...")
def get_color_features():
    return build_color_index()[1]


@st.cache_resource(show_spinner="Model yükleniyor...")
def get_model():
    return load_model()


@st.cache_resource(show_spinner="Metin arama modeli (CLIP) yükleniyor...")
def get_clip_model():
    return load_clip_model()


@st.cache_resource(show_spinner="Metin arama indeksi hazırlanıyor...")
def get_text_index():
    return build_text_index(model=get_clip_model())


@st.cache_resource(show_spinner="Etiketler hazırlanıyor...")
def get_tag_lookup():
    # önbellekler geçerliyse CLIP modeli yüklenmez
    text_paths, clip_features = build_text_index()
    return tags_by_name(text_paths, build_tag_index(text_paths, clip_features))


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


def find_similar(image_path, modifier, top_n, text_weight, color_weight):
    """Değişiklik metni yoksa VGG16 (+ isteğe bağlı renk) ile, varsa CLIP ile görsel + metin birlikte arar."""
    if not modifier:
        # veri setindeki görselin vektörü önbellekte: model yalnızca yüklenen görsel için açılır
        model = None if image_path in paths else get_model()
        color_features = get_color_features() if color_weight > 0 else None
        return recommend(image_path, paths, features, top_n=top_n, model=model,
                         color_features=color_features, color_weight=color_weight)
    text_paths, text_features = get_text_index()
    return combined_search(image_path, modifier, text_paths, text_features, top_n=top_n,
                           text_weight=text_weight, model=get_clip_model())


def refine(results):
    """Etiket filtrelerini uygular ve ilk `top_n` sonucu döndürür."""
    filtered = filter_results(results, tag_lookup, selected_tags)
    if results and not filtered:
        st.warning("Seçili filtrelere uyan sonuç yok.")
    return filtered[:top_n]


def show_grid(results, combined=False):
    for start in range(0, len(results), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (rank, (path, score)) in zip(cols, enumerate(results[start:start + GRID_COLUMNS], start=start + 1)):
            with col:
                st.image(str(path), width="stretch")
                st.markdown(f"**{rank}.** {label(path)}")
                st.caption(" · ".join(tag_lookup[path.name].values()))
                if combined:
                    # z-skorlarının ağırlıklı toplamı: 0–1 aralığında değil, yalnızca sıralama için
                    st.caption(f"Uyum skoru: {score:+.2f}")
                else:
                    st.progress(max(0.0, min(score, 1.0)), text=f"Benzerlik: {score:.3f}")


st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")
st.title("👗 Fashion Recommender")
st.caption("Bir kıyafet seçin, yükleyin veya tarif edin; en çok benzeyen kıyafetleri önerelim.")

paths, features = get_index()
tag_lookup = get_tag_lookup()

with st.sidebar:
    st.header("Ayarlar")
    source = st.radio("Arama şekli", ["Veri setinden seç", "Görsel yükle", "Metinle ara"])
    top_n = st.slider("Öneri sayısı", min_value=1, max_value=12, value=GRID_COLUMNS)
    st.subheader("Filtreler")
    selected_tags = {group: st.multiselect(group, list(labels), placeholder="Tümü")
                     for group, labels in TAG_GROUPS.items()}
    st.caption(f"Veri setinde {len(paths)} görsel var. Etiketler CLIP ile otomatik verilir ve yanılabilir "
               "(doğruluk ≈%85).")

if source == "Metinle ara":
    query = st.text_input("Aradığınız kıyafeti tarif edin (Türkçe veya İngilizce)", placeholder="kırmızı payetli elbise").strip()
    if not query:
        st.info("Başlamak için bir tarif yazın, ör. *siyah dantel elbise* veya *white trousers*.")
    else:
        text_paths, text_features = get_text_index()
        st.subheader("Sonuçlar")
        show_grid(refine(search(query, text_paths, text_features, top_n=len(paths), model=get_clip_model())))
else:
    if source == "Veri setinden seç":
        query_image = st.selectbox("Kıyafet", paths, format_func=label)
    else:
        query_image = st.file_uploader("Görsel yükle", type=[ext.lstrip(".") for ext in IMAGE_EXTENSIONS])

    modifier_col, text_col, color_col = st.columns([3, 1, 1])
    with modifier_col:
        modifier = st.text_input("Değişiklik (isteğe bağlı): bunun gibi ama…",
                                 placeholder="kırmızı, uzun kollu, payetli…").strip()
    with text_col:
        text_weight = st.slider("Metin ağırlığı", 0.0, 1.0, 0.5, 0.1, disabled=not modifier,
                                help="0 = yalnız görsel, 1 = yalnız metin")
    with color_col:
        color_weight = st.slider("Renk önemi", 0.0, 1.0, 0.0, 0.1, disabled=bool(modifier),
                                 help="Rengi benzer kıyafetleri öne çıkarır; değişiklik metni varken kullanılmaz. "
                                      "Arttıkça stil/tür benzerliği azalır.")
        if modifier:
            color_weight = 0.0

    if query_image is None:
        st.info("Başlamak için bir kıyafet görseli yükleyin.")
    else:
        with st.spinner("Benzer kıyafetler aranıyor..."):
            if isinstance(query_image, Path):
                results = refine(find_similar(query_image, modifier, len(paths), text_weight, color_weight))
            else:
                with uploaded_file(query_image) as tmp_path:
                    results = refine(find_similar(tmp_path, modifier, len(paths), text_weight, color_weight))

        left, right = st.columns([1, 3], gap="large")
        with left:
            st.subheader("Seçilen")
            st.image(query_image if not isinstance(query_image, Path) else str(query_image), width="stretch")
        with right:
            st.subheader(f"Öneriler: bunun gibi ama “{modifier}”" if modifier else "Öneriler")
            show_grid(results, combined=bool(modifier) or color_weight > 0)
