"""Streamlit web arayüzü.

    streamlit run app.py
"""

import os
import tempfile
from contextlib import contextmanager
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow log kalabalığını gizle

import streamlit as st  # noqa: E402

from fashion_rec.backbone import build_image_index, clip_encoder  # noqa: E402
from fashion_rec.calibration import combined_scale, image_scale, text_scale  # noqa: E402
from collections import Counter  # noqa: E402

from fashion_rec.catalog import display_name, load_metadata, metadata_tags  # noqa: E402
from fashion_rec.color import build_color_index  # noqa: E402
from fashion_rec.config import IMAGE_BACKBONE, IMAGE_EXTENSIONS  # noqa: E402
from fashion_rec.data import catalog_index  # noqa: E402
from fashion_rec.features import load_model  # noqa: E402
from fashion_rec.recommend import recommend  # noqa: E402
from fashion_rec.tags import TAG_GROUPS, build_tag_index, filter_results, tags_by_name  # noqa: E402
from fashion_rec.text_search import build_text_index, combined_search, load_clip_model, search  # noqa: E402

GRID_COLUMNS = 4


@st.cache_resource(show_spinner="Görsel özellikleri hazırlanıyor...")
def get_index():
    return build_image_index()


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


PICKER_LIMIT = 300  # büyük katalogda ürün seçicide gösterilecek en fazla eşleşme


@st.cache_resource(show_spinner="Etiketler hazırlanıyor...")
def get_tag_lookup():
    """Veri setinin kendi etiketleri varsa onları, yoksa CLIP'in otomatik etiketlerini kullanır."""
    metadata = get_metadata()
    if metadata:
        return metadata_tags(paths, metadata), False
    # önbellekler geçerliyse CLIP modeli yüklenmez
    text_paths, clip_features = build_text_index()
    return tags_by_name(text_paths, build_tag_index(text_paths, clip_features)), True


@st.cache_resource
def get_filter_options():
    """Grup -> etiketler, en sık görülenden en aza (büyük katalogda 141 ürün türü gibi uzun listeler için)."""
    groups = next(iter(tag_lookup.values())).keys()
    return {g: [label for label, _ in Counter(t[g] for t in tag_lookup.values()).most_common()] for g in groups}


def product_picker():
    """Küçük katalogda açılır liste; büyük katalogda önce ad araması, sonra filtrelere uyan ilk eşleşmeler."""
    if len(paths) <= PICKER_LIMIT:
        return st.selectbox("Kıyafet", paths, format_func=label)
    words = st.text_input("Ürün adında ara", placeholder="ör. navy blue shirt, titan watch").lower().split()
    # her kelime adda geçmeli, sıra önemsiz ("titan watch" → "Titan Women Silver Watch")
    candidates = [p for p in paths if all(w in label(p).lower() for w in words)] if words else paths
    candidates = [p for p, _ in filter_results([(p, 0.0) for p in candidates], tag_lookup, selected_tags)]
    if not candidates:
        st.warning("Aramaya ve filtrelere uyan ürün yok.")
        return None
    shown = candidates[:PICKER_LIMIT]
    return st.selectbox(f"Ürün ({len(candidates)} eşleşme" + (f", ilk {len(shown)} gösteriliyor)" if len(candidates) > len(shown) else ")"),
                        shown, format_func=label)


# Yüzde referansı için genel tarifler: metin araması etiket istemlerini, görsel + metin kısa değişiklikleri kullanır
REFERENCE_QUERIES = [prompt for labels in TAG_GROUPS.values() for prompt in labels.values()]
REFERENCE_MODIFIERS = ["kırmızı", "siyah", "beyaz", "mavi", "uzun kollu", "payetli"]


@st.cache_resource(show_spinner="Benzerlik ölçeği hazırlanıyor...")
def get_image_scale(color_weight):
    color_features = get_color_features() if color_weight > 0 else None
    return image_scale(paths, features, color_features=color_features, color_weight=color_weight)


@st.cache_resource(show_spinner="Benzerlik ölçeği hazırlanıyor...")
def get_text_scale():
    text_paths, text_features = get_text_index()
    return text_scale(text_paths, text_features, get_clip_model(), REFERENCE_QUERIES)


@st.cache_resource(show_spinner="Benzerlik ölçeği hazırlanıyor...")
def get_combined_scale(text_weight):
    text_paths, text_features = get_text_index()
    return combined_scale(text_paths, text_features, get_clip_model(), REFERENCE_MODIFIERS,
                          text_weight=text_weight, sample=10)


@st.cache_resource
def get_metadata():
    return load_metadata()


def label(path: Path) -> str:
    return display_name(path, get_metadata())


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
    """Değişiklik metni yoksa görsel modeliyle (+ isteğe bağlı renk), varsa CLIP ile görsel + metin birlikte arar."""
    if not modifier:
        # veri setindeki görselin vektörü önbellekte: model yalnızca yüklenen görsel için açılır
        model = encode = None
        if image_path not in catalog_index(paths)[1]:
            if IMAGE_BACKBONE == "clip":
                encode = clip_encoder(get_clip_model())
            else:
                model = get_model()
        color_features = get_color_features() if color_weight > 0 else None
        return recommend(image_path, paths, features, top_n=top_n, model=model, encode=encode,
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


def percent_text(pct):
    return f"%{pct:.1f}".replace(".", ",")


def show_grid(results, scale):
    """Sonuçları ızgara halinde, benzerliği `scale` ile yüzdeye çevirerek gösterir."""
    for start in range(0, len(results), GRID_COLUMNS):
        cols = st.columns(GRID_COLUMNS)
        for col, (rank, (path, score)) in zip(cols, enumerate(results[start:start + GRID_COLUMNS], start=start + 1)):
            with col:
                st.image(str(path), width="stretch")
                st.markdown(f"**{rank}.** {label(path)}")
                st.caption(" · ".join(tag_lookup[path.name].values()))
                pct = scale(score)
                st.progress(pct / 100, text=f"Benzerlik: {percent_text(pct)}")


st.set_page_config(page_title="Fashion Recommender", page_icon="👗", layout="wide")
st.title("👗 Fashion Recommender")
st.caption("Bir kıyafet seçin, yükleyin veya tarif edin; en çok benzeyen kıyafetleri önerelim.")

paths, features = get_index()
tag_lookup, automatic_tags = get_tag_lookup()

with st.sidebar:
    st.header("Ayarlar")
    source = st.radio("Arama şekli", ["Veri setinden seç", "Görsel yükle", "Metinle ara"])
    top_n = st.slider("Öneri sayısı", min_value=1, max_value=12, value=GRID_COLUMNS)
    st.subheader("Filtreler")
    selected_tags = {group: st.multiselect(group, labels, placeholder="Tümü")
                     for group, labels in get_filter_options().items()}
    st.caption(f"Veri setinde {len(paths)} görsel var. " +
               ("Etiketler CLIP ile otomatik verilir ve yanılabilir (doğruluk ≈%85)." if automatic_tags
                else "Etiketler veri setinin kendi ürün bilgilerinden gelir."))
    st.caption("**Benzerlik %** tipik en iyi eşleşmelerle karşılaştırmadır: %80, örnek aramaların ilk 10 "
               "sonucunun %80'inden daha benzer demektir.")

if source == "Metinle ara":
    query = st.text_input("Aradığınız kıyafeti tarif edin (Türkçe veya İngilizce)", placeholder="kırmızı payetli elbise").strip()
    if not query:
        st.info("Başlamak için bir tarif yazın, ör. *siyah dantel elbise* veya *white trousers*.")
    else:
        text_paths, text_features = get_text_index()
        st.subheader("Sonuçlar")
        show_grid(refine(search(query, text_paths, text_features, top_n=len(paths), model=get_clip_model())),
                  get_text_scale())
else:
    if source == "Veri setinden seç":
        query_image = product_picker()
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
        if source == "Görsel yükle":
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
            scale = get_combined_scale(text_weight) if modifier else get_image_scale(color_weight)
            show_grid(results, scale)
