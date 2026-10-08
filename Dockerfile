# Fashion Recommender — Streamlit arayüzü
#
#   docker compose build
#   docker compose run --rm app python scripts/setup_data.py hm   # veri setini indir ve hazırla
#   docker compose up                                              # http://localhost:8501
#
# Görseller ve önbellekler imaja konmaz (data/, cache/ birim olarak bağlanır); model ağırlıkları
# (CLIP ~1,5 GB, VGG16 ~60 MB) ilk kullanımda /models birimine iner.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/models/huggingface \
    KERAS_HOME=/models/keras \
    TF_CPP_MIN_LOG_LEVEL=3 \
    TOKENIZERS_PARALLELISM=false

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# PyTorch'un CPU sürümü: CUDA'lı sürüm imajı birkaç GB büyütür, Docker'da GPU yok
COPY requirements.txt .
RUN pip install --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY . .

EXPOSE 8501
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD curl -fs http://localhost:8501/_stcore/health || exit 1
CMD ["streamlit", "run", "app.py", "--server.address=0.0.0.0", "--server.port=8501", "--server.headless=true"]
