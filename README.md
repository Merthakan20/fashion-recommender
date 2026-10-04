# 👗 Fashion Recommender

Görsel benzerliğine dayalı kıyafet öneri sistemi. Bir kıyafet görseli verildiğinde, veri setindeki görsel olarak en çok benzeyen kıyafetleri önerir.

Üç şekilde kullanılabilir:

- **Python paketi** — `fashion_rec`
- **Komut satırı (CLI)** — `python -m fashion_rec ...`
- **Web arayüzü** — Streamlit

![Streamlit arayüzü](docs/screenshot.png)

## Nasıl çalışır

1. `data/women fashion/` altındaki görseller listelenir (`.jpg`, `.jpeg`, `.png`, `.webp`).
2. ImageNet üzerinde önceden eğitilmiş **VGG16** ile her görselden bir özellik vektörü (25.088 boyut) çıkarılır ve L2 ile normalize edilir.
3. Vektörler `features.npz` dosyasına önbelleklenir. Görsel listesi değişince otomatik olarak yeniden hesaplanır.
4. Girdi görselinin vektörü tüm vektörlerle çarpılır (normalize vektörlerde nokta çarpımı = cosine benzerliği) ve en yüksek skorlu N görsel döner. Girdi görseli veri setindeyse sonuçlardan çıkarılır.

Veri setindeki bir görsel için vektör önbellekten alınır, model hiç yüklenmez. Model yalnızca veri setinde olmayan bir görsel geldiğinde açılır.

## Kurulum

Python 3.11 gerekir. Örnek (conda):

```bash
conda create -n fashion-rec python=3.11
conda activate fashion-rec
pip install -r requirements.txt
```

İlk çalıştırmada VGG16 ağırlıkları (~60 MB) Keras tarafından otomatik indirilir.

## Kullanım

Tüm komutlar proje klasöründen çalıştırılır.

### Komut satırı

Özellikleri çıkarıp önbelleğe almak için (96 görsel için birkaç saniye sürer):

```bash
python -m fashion_rec index           # önbellek varsa onu kullanır
python -m fashion_rec index --force   # önbelleği yok sayıp yeniden hesaplar
```

Bir görsele benzeyen kıyafetleri önermek için:

```bash
python -m fashion_rec recommend "data/women fashion/black floral saree.jpg" -n 5
```

```
Girdi: black floral saree.jpg

 1. 0.455  Anarkali suit with fitted bodice with a high neckline.jpg
 2. 0.412  long, intricately designed dress with full sleeves.jpg
 3. 0.393  anarkali suit with intricate silver embellishments on the neckline, sleeves.jpg
 4. 0.379  anarkali suit with lavender in color with intricate white patterns throughout the fabric.jpg
 5. 0.376  black jumpsuit with a diagonal stripe of glittering material running across the body.jpg
```

Girdi görseli veri setinde olmak zorunda değil, herhangi bir görsel yolu verilebilir. Önbellek yoksa `recommend` onu önce kendisi oluşturur.

### Web arayüzü

```bash
streamlit run app.py
```

Kenar çubuğundan görsel kaynağını (veri setinden seç / görsel yükle) ve öneri sayısını (1–12) seçin. Öneriler benzerlik skorlarıyla birlikte ızgara halinde gösterilir.

### Python'dan

```python
from fashion_rec.features import build_index
from fashion_rec.recommend import recommend

paths, features = build_index()
for path, score in recommend("benim_elbisem.jpg", paths, features, top_n=5):
    print(f"{score:.3f}  {path.name}")
```

## Testler

```bash
python -m pytest
```

Testler sahte, hafif bir model ve düz renkli test görselleri kullanır. TensorFlow yüklenmeden saniyenin altında çalışırlar.

## Proje yapısı

```
fashion-recommender/
├── fashion_rec/          # Python paketi
│   ├── __main__.py       # CLI giriş noktası
│   ├── config.py         # yollar, görsel boyutu, batch boyutu
│   ├── data.py           # görsel listeleme
│   ├── features.py       # model yükleme, özellik çıkarma, önbellek
│   └── recommend.py      # benzerlik hesabı + öneri
├── app.py                # Streamlit arayüzü
├── data/women fashion/   # 96 kadın kıyafeti görseli
├── docs/                 # README görselleri
├── tests/                # pytest testleri
├── requirements.txt
└── PROJE.md              # yol haritası ve çalışma notları
```

Ayarlar (veri klasörü, önbellek yolu, görsel boyutu, batch boyutu) `fashion_rec/config.py` içindedir.

## Veri seti

96 kadın kıyafeti görseli. Dosya adları kıyafetin açıklamasıdır (ör. `black floral saree.jpg`) ve arayüzde etiket olarak kullanılır.
