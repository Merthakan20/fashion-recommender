# Fashion Recommender — Yol Haritası

Görsel benzerliğine dayalı kıyafet öneri sistemi. Proje sıfırdan, adım adım birlikte yazılıyor. Her aşama bitince kutucuğu işaretleyip **Çalışma notları** bölümüne kısa bir not ekliyoruz.

> Eski deneme notebook'u: `../Fashion Recommendation/Fashion Recommendation.ipynb` (yalnızca referans; bu projede kullanılmıyor).

## 1. Amaç

Bir kıyafet görseli verildiğinde, veri setindeki en çok benzeyen kıyafetleri önermek:

- yeniden kullanılabilir bir **Python paketi** (`fashion_rec`),
- komut satırından kullanılabilir bir **CLI**,
- tarayıcıdan kullanılabilir bir **web arayüzü** (Streamlit).

## 2. Nasıl çalışacak

1. `data/women fashion/` altındaki görseller listelenir (jpg, jpeg, png, webp).
2. Önceden eğitilmiş bir CNN (başlangıçta VGG16, ImageNet) ile her görselden bir özellik vektörü çıkarılır ve L2 ile normalize edilir.
3. Vektörler `features.npz` dosyasına önbelleklenir; görsel listesi değişince yeniden hesaplanır.
4. Girdi görselinin vektörüyle tüm vektörlerin nokta çarpımı alınır (= cosine benzerliği), en yüksek N sonuç döner. Girdi görselinin kendisi sonuçlardan çıkarılır.

**Veri seti:** 96 kadın kıyafeti görseli. Dosya adları kıyafetin açıklaması (ör. `black floral saree.jpg`).

## 3. Klasör yapısı

```
fashion-recommender/
├── fashion_rec/              # Python paketi
│   ├── __init__.py
│   ├── __main__.py           # CLI giriş noktası
│   ├── config.py             # yollar, model ayarları
│   ├── data.py               # görsel listeleme
│   ├── features.py           # model yükleme, özellik çıkarma, önbellek
│   └── recommend.py          # benzerlik hesabı + öneri
├── app.py                    # Streamlit arayüzü
├── notebooks/                # deneme / görselleştirme notebook'ları
├── data/
│   └── women fashion/        # görseller (96 adet)
├── tests/
├── requirements.txt
├── .gitignore
├── README.md
└── PROJE.md                  # bu dosya
```

## 4. Aşamalar

### Aşama 0 — Kurulum
- [x] Yeni klasörü ve alt klasörleri oluştur
- [x] Görselleri `data/women fashion/` altına kopyala
- [x] `requirements.txt` ve `.gitignore` yaz
- [x] Proje için conda ortamı oluştur (`conda create -n fashion-rec python=3.11`)
- [x] Bağımlılıkları kur (`pip install -r requirements.txt`) — sürümler sabitlendi; TensorFlow 2.21 ve VGG16 ağırlıkları doğrulandı
- [x] `git init` ve ilk commit

### Aşama 1 — Paket
- [ ] `config.py`: veri klasörü, önbellek yolu, model adı, görsel boyutu
- [ ] `data.py`: `list_images()` — desteklenen uzantılar, büyük/küçük harf duyarsız, sıralı
- [ ] `features.py`: `load_model()`, `extract_features(path)`, önbellekli `build_index()`
- [ ] `recommend.py`: `recommend(image_path, top_n) -> [(yol, skor), ...]`

### Aşama 2 — Komut satırı (CLI)
- [ ] `python -m fashion_rec index` → özellikleri çıkar ve önbellekle
- [ ] `python -m fashion_rec recommend <görsel> -n 5` → önerileri skorlarıyla yazdır

### Aşama 3 — Testler
- [ ] Sahte (hafif) bir modelle pytest altyapısı (TensorFlow gerektirmeden çalışsın)
- [ ] Önbellek: ikinci çalıştırmada model çağrılmıyor; görsel listesi değişince yeniden hesaplanıyor
- [ ] Öneri: girdi görseli sonuçlarda yok, sonuçlar azalan skorla sıralı, tam `top_n` sonuç dönüyor
- [ ] Veri setinde olmayan bir görselle çalışıyor

### Aşama 4 — Web arayüzü (Streamlit)
- [ ] Görsel yükleme **veya** veri setinden görsel seçme
- [ ] Öneri sayısı için kaydırıcı (top-N)
- [ ] Önerileri benzerlik skorlarıyla ızgara halinde gösterme
- [ ] Model ve özellikleri `st.cache_resource` ile bir kez yükleme

### Aşama 5 — Model iyileştirmeleri (isteğe bağlı)
- [ ] `pooling='avg'` ile vektörü 25.088 → 512 boyuta indir; kaliteyi karşılaştır
- [ ] ResNet50 / EfficientNet / CLIP karşılaştırması
- [ ] CLIP ile metinden arama (ör. "kırmızı payetli elbise")
- [ ] Birkaç örnek görsel için önerileri yan yana kaydedip gözle değerlendirme

### Aşama 6 — Dokümantasyon
- [ ] `README.md`: proje tanımı, kurulum, CLI ve arayüz kullanımı, ekran görüntüsü

## 5. Açık sorular / kararlar

| Konu | Seçenekler | Öneri | Karar |
|---|---|---|---|
| Web arayüzü | Streamlit / Gradio | Streamlit | Streamlit |
| Ortam | conda / venv | conda (Anaconda zaten kurulu) | conda (`fashion-rec`) |
| Model | VGG16 / ResNet50 / CLIP | Önce VGG16, Aşama 5'te karşılaştır | — |
| Sürüm kontrolü | git var / yok | git | git |

## 6. Çalışma notları

- **2026-10-04** — Proje sıfırdan `fashion-recommender/` klasöründe başlatıldı. Klasör yapısı, veri, `requirements.txt` ve `.gitignore` hazır. Sıradaki adım: conda ortamı + bağımlılık kurulumu.
- **2026-10-04** — Aşama 0 tamamlandı: `fashion-rec` conda ortamı (Python 3.11) kuruldu, bağımlılıklar yüklendi ve sabitlendi, git başlatıldı. Ortamı açmak için: `conda activate fashion-rec`. Sıradaki adım: **Aşama 1 — Paket**.
