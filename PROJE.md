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

**Veri seti:** 93 kadın kıyafeti görseli (3 kopya ayıklandı). Dosya adları kıyafetin açıklaması (ör. `black floral saree.jpg`).

## 3. Klasör yapısı

```
fashion-recommender/
├── fashion_rec/              # Python paketi
│   ├── __init__.py
│   ├── __main__.py           # CLI giriş noktası
│   ├── config.py             # yollar, model ayarları
│   ├── data.py               # görsel listeleme
│   ├── features.py           # model yükleme, özellik çıkarma, önbellek
│   ├── recommend.py          # benzerlik hesabı + öneri
│   ├── text_search.py        # CLIP ile metinden arama
│   ├── tags.py               # otomatik renk / tür etiketleri + filtre
│   ├── calibration.py        # benzerlik yüzdesi
│   ├── catalog.py            # ürün adları + metadata etiketleri
│   ├── backbone.py           # görsel öneri modeli seçimi
│   ├── adapter.py            # CLIP adaptörü (isteğe bağlı)
│   ├── glossary.py           # Türkçe moda sözlüğü
│   ├── diversity.py          # çeşitlilik (MMR)
│   ├── outfit.py             # kombin önerisi
│   ├── copurchase.py         # birlikte alınanlar
│   └── segment_settings.*    # bölüm bazında ayarlar
├── app.py                    # Streamlit arayüzü
├── notebooks/                # deneme / görselleştirme notebook'ları
├── data/
│   └── women fashion/        # görseller (93 adet)
├── docs/                     # README görselleri
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
- [x] `config.py`: veri klasörü, önbellek yolu, görsel boyutu, batch boyutu
- [x] `data.py`: `list_images()` — desteklenen uzantılar, büyük/küçük harf duyarsız, sıralı
- [x] `features.py`: `load_model()`, `extract_features(model, paths)` (batch'li), önbellekli `build_index()`
- [x] `recommend.py`: `recommend(image_path, paths, features, top_n) -> [(yol, skor), ...]`

### Aşama 2 — Komut satırı (CLI)
- [x] `python -m fashion_rec index [--force]` → özellikleri çıkar ve önbellekle
- [x] `python -m fashion_rec recommend <görsel> -n 5` → önerileri skorlarıyla yazdır

### Aşama 3 — Testler
- [x] Sahte (hafif) bir modelle pytest altyapısı (TensorFlow gerektirmeden çalışsın)
- [x] Önbellek: ikinci çalıştırmada model çağrılmıyor; görsel listesi değişince yeniden hesaplanıyor
- [x] Öneri: girdi görseli sonuçlarda yok, sonuçlar azalan skorla sıralı, tam `top_n` sonuç dönüyor
- [x] Veri setinde olmayan bir görselle çalışıyor
- [x] (Ek) `data.py` ve CLI testleri

### Aşama 4 — Web arayüzü (Streamlit)
- [x] Görsel yükleme **veya** veri setinden görsel seçme
- [x] Öneri sayısı için kaydırıcı (top-N, 1–12)
- [x] Önerileri benzerlik skorlarıyla ızgara halinde gösterme
- [x] Model ve özellikleri `st.cache_resource` ile bir kez yükleme (model yalnızca görsel yüklenince açılıyor)
- [x] Çalıştırma: `streamlit run app.py`

### Aşama 5 — Model iyileştirmeleri (isteğe bağlı)
- [x] `pooling='avg'` ile vektörü 25.088 → 512 boyuta indir; kaliteyi karşılaştır → belirgin kazanç yok, **flatten kalıyor** (`notebooks/compare_pooling.py`)
- [x] ResNet50 / EfficientNet / CLIP karşılaştırması → CLIP (ViT-B/32, ViT-B/16) denendi, görselden görsele net üstünlük yok; ResNet/EfficientNet atlandı (`notebooks/compare_models.py`)
- [x] CLIP ile metinden arama (ör. "red sequined dress") → `fashion_rec/text_search.py`, CLI `search`, arayüzde "Metinle ara"
- [x] Birkaç örnek görsel için önerileri yan yana kaydedip gözle değerlendirme (`notebooks/pooling_comparison/`; yeni modeller eklendikçe genişletilecek)

### Aşama 6 — Dokümantasyon
- [x] `README.md`: proje tanımı, kurulum, CLI ve arayüz kullanımı, ekran görüntüsü (`docs/screenshot.png`)

### Aşama 7 — Yeni özellikler
- [x] Görsel + metin birlikte arama ("bunun gibi ama kırmızı") → `combined_search`, CLI `combine`, arayüzde "Değişiklik" kutusu
- [x] Renk duyarlı benzerlik: orta bölge HSV histogramı + "renk önemi" kaydırıcısı (varsayılan 0) → `fashion_rec/color.py`, `recommend(color_weight=…)`, CLI `-c`
- [x] Otomatik etiketler ve filtreler: CLIP ile renk / tür etiketleri, kenar çubuğunda filtre → `fashion_rec/tags.py`, CLI `tags`

## 5. Açık sorular / kararlar

| Konu | Seçenekler | Öneri | Karar |
|---|---|---|---|
| Web arayüzü | Streamlit / Gradio | Streamlit | Streamlit |
| Ortam | conda / venv | conda (Anaconda zaten kurulu) | conda (`fashion-rec`) |
| Model | VGG16 / ResNet50 / CLIP | Önce VGG16, Aşama 5'te karşılaştır | Görselden görsele: VGG16 flatten. Metinden arama: çok dilli CLIP `xlm-roberta-base-ViT-B-32` (LAION), Türkçe + İngilizce |
| Kopya görseller | sil / bırak | — | Düşük çözünürlüklü kopya silindi, yüksek çözünürlüklü dosyaya açıklayıcı ad verildi |
| Sürüm kontrolü | git var / yok | git | git |

## 6. Çalışma notları

- **2026-10-04** — Proje sıfırdan `fashion-recommender/` klasöründe başlatıldı. Klasör yapısı, veri, `requirements.txt` ve `.gitignore` hazır. Sıradaki adım: conda ortamı + bağımlılık kurulumu.
- **2026-10-04** — Aşama 0 tamamlandı: `fashion-rec` conda ortamı (Python 3.11) kuruldu, bağımlılıklar yüklendi ve sabitlendi, git başlatıldı. Ortamı açmak için: `conda activate fashion-rec`. Sıradaki adım: **Aşama 1 — Paket**.
- **2026-10-04** — Aşama 1 tamamlandı: `fashion_rec` paketi yazıldı ve gerçek VGG16 ile denendi. 96 görselin özellik çıkarımı ~7 sn, önbellekten yükleme anında. Veri setindeki görseller için model hiç yüklenmiyor (vektör önbellekten alınıyor). Sıradaki adım: **Aşama 2 — CLI**.
- **2026-10-04** — Aşama 2 tamamlandı: `fashion_rec/__main__.py` ile `index` ve `recommend` komutları eklendi. Hatalı girdiler (olmayan dosya, `-n 0`) anlaşılır mesajla reddediliyor. Komutlar proje klasöründen, `fashion-rec` ortamı açıkken çalıştırılır. Sıradaki adım: **Aşama 3 — Testler**.
- **2026-10-04** — Aşama 3 tamamlandı: `tests/` altında 28 test (data, features, recommend, CLI). TensorFlow yüklenmeden ~0,1 sn'de çalışıyor; düz renkli test görselleri ve sahte model kullanılıyor. Koda bilerek eklenen iki hata (ters sıralama, girdi görselinin çıkarılmaması) testlerce yakalandı. Çalıştırma: `python -m pytest`. Sıradaki adım: **Aşama 4 — Web arayüzü**.
- **2026-10-04** — Aşama 1–3 commit edildi (`e24575f`). Aşama 4 tamamlandı: `app.py` Streamlit arayüzü. AppTest ile veri setinden seçim, kaydırıcı ve mod değişimi; gerçek modelle yükleme yolu denendi (küçültülüp PNG yapılan sari, aslını 0,971 skorla buldu; geçici dosya siliniyor). Sıradaki adım: **Aşama 5 (isteğe bağlı) veya Aşama 6 — README**.
- **2026-10-04** — Aşama 6 tamamlandı: `README.md` yazıldı (nasıl çalışır, kurulum, CLI / arayüz / Python kullanımı, testler, proje yapısı). CLI örnek çıktısı gerçek çalıştırmadan alındı; arayüz ekran görüntüsü `docs/screenshot.png`. 28 test geçiyor. Sıradaki adım: **Aşama 5 (isteğe bağlı) — model iyileştirmeleri**.
- **2026-10-04** — Aşama 6 commit edildi. Aşama 5'e başlandı: `notebooks/compare_pooling.py` ile VGG16 çıktısı için flatten (25.088), avg (512) ve max (512) karşılaştırıldı. Etiket olmadığından dosya adlarından ölçüt çıkarıldı:

  | yöntem | etnik@5 | renk@5 | eş sırası (düşük = iyi) |
  |---|---|---|---|
  | flatten | 0,49 | 0,17 | 4,6 |
  | avg | 0,49 | 0,20 | 6,8 |
  | max | 0,47 | 0,20 | 3,1 |

  Farklar küçük ve örnek az (11 etnik sorgu, 8 eş sırası); yöntemler ilk 5'te ortalama yalnızca ~2 görselde anlaşıyor. Gözle bakınca flatten payet/doku benzerliğinde (ör. "sparkling black dress") açıkça daha iyi; havuzlama konumu kaybettiği için desenli ama alakasız elbiseler getiriyor. 96 görselde boyut (9,6 MB → 0,2 MB) önemsiz. **Karar: flatten kalıyor.** Üç yöntem de renk tutarlılığında zayıf (≈0,2) — asıl iyileşme için CLIP denenmeli.
  Benzerlikle bulunan neredeyse aynı görseller (cosine > 0,8): `blue dress with a floral pattern` ↔ `l-htdrss7133-honky-tonky…` (0,92), `Sleeveless-Ruched-…-Club-Dresses` ↔ `fitted, off-the-shoulder white dress with horizontal ribbed texture` (0,91), `Latest-Party-Wear-One-Piece-Gown…` ↔ `long, flowing dress with a full skirt` (0,83). Sıradaki adım: **CLIP karşılaştırması**.
- **2026-10-04** — Havuzlama karşılaştırması commit edildi (`88a3be1`). `open_clip_torch` 3.3.0 (torch 2.14.1) ortama kuruldu (henüz `requirements.txt`'te değil). `notebooks/compare_models.py` ile VGG16 ve CLIP karşılaştırıldı:

  | yöntem | boyut | etnik@5 | renk@5 | eş sırası |
  |---|---|---|---|---|
  | vgg16 (flatten) | 25.088 | 0,49 | 0,17 | 4,6 |
  | clip ViT-B/32 | 512 | 0,56 | 0,19 | 4,2 |
  | clip ViT-B/16 | 512 | 0,64 | 0,19 | 7,2 |

  Görselden görsele: belirgin kazanan yok. CLIP kombin/stil türünü daha iyi yakalıyor ("black top, white trousers" → blazer, takım, sokak stili), VGG16 doku ve deseni daha iyi yakalıyor ("sparkling black dress" → payetli elbiseler; "black floral saree" → etnik kıyafetler). Renk tutarlılığı ikisinde de zayıf.
  Metinden arama (CLIP): çok iyi — "a red sequined dress", "a houndstooth dress", "white trousers", "a black lace dress" sorgularında doğru görseller ilk 3'te. B/32 ve B/16 benzer; B/32 daha hızlı ve küçük.
  **Öneri:** görselden görsele VGG16 kalsın, CLIP ViT-B/32 metinden arama için eklensin. Sıradaki adım: **metinden aramayı pakete / CLI'ye / arayüze ekleme**.
- **2026-10-04** — CLIP karşılaştırması commit edildi (`a43fb18`). Metinden arama eklendi: `fashion_rec/text_search.py` (`ClipModel`, `build_text_index`, `search`), ayrı önbellek `clip_features.npz`. Önbellek mantığı `features.cached_features()` olarak ortaklaştırıldı. CLI: `search "<metin>" -n 5` ve `index --text`. Arayüz: "Metinle ara" modu (CLIP modeli ve indeksi `st.cache_resource` ile, yalnızca bu mod seçilince yükleniyor). `torch`, `torchvision`, `open_clip_torch` `requirements.txt`'e zorunlu olarak eklendi. TensorFlow ve torch'un aynı süreçte birlikte çalıştığı doğrulandı. Testler 38'e çıktı (sahte CLIP modeli: renk kelimesi → RGB); bilerek ters çevrilen sıralama 4 testte yakalandı. Arayüz tarayıcıda denendi ("black lace dress" → siyah dantel/payetli elbiseler). Aşama 5'in tüm maddeleri tamamlandı.
- **2026-10-04** — Metinden arama commit edildi (`b0e0829`). Kopya görseller ayıklandı (96 → 93): her çiftte anlamsız adlı dosya aslında yüksek çözünürlüklüydü (ör. 488×650'ye karşı 194×259), bu yüzden düşük çözünürlüklü kopya silindi ve yüksek çözünürlüklü dosyaya açıklayıcı ad verildi (`blue dress with a floral pattern.jpeg`, `fitted, off-the-shoulder white dress with horizontal ribbed texture.webp`, `long, flowing dress with a full skirt.webp`). İndeksler yeniden oluşturuldu; VGG16'da en yüksek benzerlik artık 0,70 (aynı kıyafetin farklı pozu). `tight-fitting, off-the-shoulder white dress` ve `… 2` aynı elbisenin iki farklı fotoğrafı olduğu için bırakıldı. Sıradaki adım: **çok dilli CLIP ile Türkçe arama denemesi**.
- **2026-10-04** — Kopyalar commit edildi (`1460c31`). Türkçe arama için çok dilli modeller `notebooks/compare_multilingual.py` ile karşılaştırıldı (20 Türkçe/İngilizce sorgu çifti, doğru cevap dosya adındaki kelimelerle tanımlı):

  | model | dil | hit@1 | hit@3 | MRR |
  |---|---|---|---|---|
  | OpenAI CLIP ViT-B/32 (eski) | tr | 0,05 | 0,15 | 0,16 |
  | OpenAI CLIP ViT-B/32 (eski) | en | 0,90 | 1,00 | 0,93 |
  | `clip-ViT-B-32-multilingual-v1` (sentence-transformers) | tr | 0,30 | 0,45 | 0,41 |
  | `clip-ViT-B-32-multilingual-v1` (sentence-transformers) | en | 0,40 | 0,75 | 0,57 |
  | **`xlm-roberta-base-ViT-B-32` (LAION)** | tr | **0,70** | **0,95** | **0,82** |
  | **`xlm-roberta-base-ViT-B-32` (LAION)** | en | **0,95** | **1,00** | **0,97** |

  **Karar: LAION xlm-roberta modeline geçildi** — Türkçede büyük sıçrama, İngilizcede kayıp yok. Bedeli: ağırlıklar 605 MB → 1,46 GB, `transformers` yeni bağımlılık. Önbelleğe model kimliği (`key`) eklendi: iki model de 512 boyutlu olduğundan, eskiden model değişse bile eski vektörler sessizce kullanılırdı (2 yeni test, toplam 40). Arayüz ve CLI metinleri "Türkçe veya İngilizce" olarak güncellendi, README'ye metinden arama ekran görüntüsü eklendi. Not: sorgu seti küçük ve dosya adlarına dayalı; sonuçlar yön gösterici.
- **2026-10-05** — Çok dilli CLIP commit edildi (`e10f1ef`). Aşama 7 açıldı; sıra: görsel + metin arama → renk duyarlı benzerlik → etiketler/filtreler. **Görsel + metin arama** eklendi. `notebooks/compare_combined.py` ile 14 örnekte (ör. `black, sequined dress` + "kırmızı" → `bright red, sequined dress`) yöntemler karşılaştırıldı:

  | yöntem | hit@1 | hit@3 | MRR |
  |---|---|---|---|
  | yalnız görsel | 0,21 | 0,29 | 0,32 |
  | yalnız metin | 0,29 | 0,57 | 0,49 |
  | vektör toplamı, w=3 | 0,50 | 0,93 | 0,70 |
  | **skor birleştirme (z-skoru), a=0,5** | 0,50 | 0,86 | 0,70 |

  Vektör toplamı ile skor birleştirme berabere; skor birleştirme seçildi çünkü ağırlık 0–1 arası anlaşılır bir kaydırıcı (0 = yalnız görsel, 1 = yalnız metin) ve en iyisi tam ortada. `text_search.combined_search()`, CLI `combine <görsel> "<metin>" [-n] [-w]`, arayüzde görsel modlarına "Değişiklik (isteğe bağlı)" kutusu + metin ağırlığı kaydırıcısı; kutu boşsa eskisi gibi VGG16 kullanılır ve veri setinden seçimde model yine yüklenmez. Testler 53'e çıktı. Birleştirmeyi gerçekten sınayan iki test var: macenta, ne görselde (turuncu) ne metinde (mavi) birinci olduğu halde birleşik aramada birinci geliyor; metin benzerlikleri yapay olarak sıkıştırıldığında da z-skoru dengeyi koruyor. Ters ağırlık ve z-skorunun kaldırılması testlerce yakalandı. Not: örnek seti küçük; `sparkling black dress` + "kırmızı" hâlâ zayıf (doğru cevap 18. sırada). Sıradaki adım: **renk duyarlı benzerlik**.
- **2026-10-05** — Görsel + metin arama commit edildi (`5c2841a`). **Renk duyarlı benzerlik** eklendi. `notebooks/compare_color.py` ile HSV renk histogramı (12×3×3, karekök + L2) VGG16 ile z-skoru + ağırlıkla birleştirildi:

  | yöntem | renk@5 | etnik@5 | eş sırası |
  |---|---|---|---|
  | vgg16 | 0,17 | 0,51 | 5,8 |
  | yalnız renk (orta bölge) | 0,37 | 0,04 | 29,8 |
  | vgg + renk (orta), c=0,3 | 0,23 | 0,38 | 5,2 |
  | vgg + renk (orta), c=0,4 | 0,26 | 0,31 | 4,8 |
  | vgg + renk (orta), c=0,5 | 0,29 | 0,22 | 5,7 |

  Orta bölge (tüm görselden iyi: 0,37'ye karşı 0,25) seçildi. Kenar şeridinden arka plan rengini tahmin edip maskelemek de denendi; ölçütleri iyileştirmedi (renk@5 benzer, eş sırası kötü), bırakıldı. Net bir ödünleşim olduğu için tek bir ağırlık seçilmedi: **"renk önemi" kaydırıcısı, varsayılan 0** (eski davranış birebir korunuyor). Gözle: "sparkling black dress" için belirgin iyileşme (siyah payetliler öne geçiyor), "red velvet sheath midi dress" için karışık (gri fon yüzünden gümüş elbise öne çıkıyor). `fashion_rec/color.py` (önbellek `color_features.npz`, anahtar histogram ayarlarını içerir), `recommend(color_features=…, color_weight=…)`, CLI `recommend -c`, arayüzde "Renk önemi" kaydırıcısı (değişiklik metni varken devre dışı). `zscore` `recommend.py`'de tek yerde. Testler 64'e çıktı; ana test: sahte VGG16 kırmızıya en yakın olarak koyu kırmızıyı, renk ağırlığı açıkken açık kırmızıyı seçiyor; birleştirme devre dışı bırakılınca test düşüyor. Sıradaki adım: **otomatik etiketler ve filtreler**.
- **2026-10-05** — Renk duyarlı benzerlik commit edildi (`7518639`). **Otomatik etiketler ve filtreler** eklendi. `notebooks/compare_tags.py` ile CLIP sıfır örnekli etiketleme dosya adlarından çıkan etiketlerle ölçüldü ve istem kalıpları karşılaştırıldı: renk 10 etiket, en iyi `"a {} outfit"` → **0,87** (60 görsel); tür 5 etiket (elbise, etnik, pantolonlu, etekli, ceket / üst), en iyi `"a photo of a woman wearing {}"` → **0,85** (89 görsel). En sık hata etekli kombinin "elbise" sayılması (4). Dağılım: elbise 65, etnik 16, pantolonlu 7, ceket / üst 3, etekli 2. `fashion_rec/tags.py` (`build_tag_index`, `filter_results`, önbellek `tags.npz`: anahtar CLIP modeli + etiket listesi; önbellek geçerliyse CLIP yüklenmez), CLI `tags [-v]`, arayüzde kenar çubuğunda Renk / Tür çoklu seçim filtreleri (üç modda da geçerli; grup içi "veya", gruplar arası "ve") ve kartlarda etiketler. Arama artık tüm adayları sıralayıp filtreden sonra ilk N'i alıyor. Testler 72'ye çıktı. Tarayıcıda denendi: "payetli elbise" + Tür=etnik → dört etnik kıyafet. **Aşama 7 tamamlandı.**
- **2026-10-05** — Etiketler commit edildi (`0a370ab`). Kategori (Erkek / Kadın / Çocuk / Aksesuar) + BigQuery fikri konuşuldu; kullanıcı isteğiyle **uygulanmadı**. **Benzerlik yüzdesi** eklendi: ham skorlar yüzde olmadığı için (CLIP metin–görsel 0,30 = çok iyi) skor, aynı yöntemle katalogda üretilen referans skorlarla karşılaştırılıyor (`fashion_rec/calibration.py`, `PercentScale`; görsel: örnek görsellerin tüm katalogla skorları, metin: etiket istemleri, görsel + metin: 10 görsel × 6 kısa değişiklik). Referans sabit olduğundan anlamsız sorgu düşük kalıyor: "siyah dantel elbise" %98–99, "uçak" %46–66. Arayüzde ilerleme çubuğu + "Benzerlik: %98,5"; CLI ham skoru göstermeye devam ediyor. Not: düşük ham skorlar model eğitimiyle "yükseltilecek" bir şey değil; eğitim sıralamayı iyileştirebilir, bunun için etiketli veri gerekir. Testler 78'e çıktı. Açık veri adayı: Fashion Product Images (Kaggle, MIT; 44 bin ürün, gender + masterCategory; HF'de 384×512 kopya, 2 GB) — görseller Myntra'dan, ticari kullanımda dikkat.
- **2026-10-05** — **Açık veri seti eklendi**: Fashion Product Images (Kaggle MIT, Myntra görselleri), HF'deki 384×512 kopya, 44.072 ürün (Erkek 22.104, Kadın 18.357, Unisex 2.126, Erkek Çocuk 830, Kız Çocuk 655; Giyim 21.361, Aksesuar 11.244, Ayakkabı 9.197…). `scripts/import_products.py` parquet → `data/products/images/<id>.jpg` + `metadata.csv` (git dışında). Veri seti `FASHION_DATASET` ile seçiliyor (`women` varsayılan, `products`), önbellekler `cache/<veri seti>/` altında (kadın önbellekleri oraya taşındı). Ölçek için: `catalog_index` (yol → sıra, liste başına bir kez), CLIP Apple GPU'da (MPS; çıktı CPU ile aynı, cosine 0,99999994), 1000+ görselde ilerleme çıktısı. 44 bin ürün: CLIP 223 sn, renk 54 sn.
  **Gerçek etiketlerle ölçüm** (`notebooks/evaluate_products.py`, 5.000 ürünlük alt katalog, 1.000 sorgu, precision@5): CLIP görsel her alanda en iyi (ürün türü 0,810 / renk 0,334 / cinsiyet 0,874) — VGG16 flatten 0,755 / 0,266 / 0,837, avg 0,716 / 0,299 / 0,810. **Karar: `products` için görsel öneri modeli CLIP** (`config.IMAGE_BACKBONE`, `fashion_rec/backbone.py`); VGG16 flatten 44 bin üründe zaten ~4,4 GB tutardı. Metinden arama ilk 10 isabet ortalaması 0,74; "kemer", "parfüm" 0 — **model eğitimi için somut hedefler**.
  Arayüz: `products`'ta filtreler metadata'dan (Cinsiyet / Kategori / Ürün türü / Renk, Türkçe), ürün seçici kelime bazlı ad araması + en fazla 300 eşleşme. **Yüzde ölçeği düzeltildi**: tüm çiftlerle kıyas 44 bin üründe her sonucu %99,9+ gösteriyordu; referans artık örnek aramaların ilk 10 sonucu (`calibration.TOP_K`). Bu iki veri setinde de tutarlı: tipik sorgunun 1. sonucu medyanda %72–78, 4. sonucu %45–53; "uçak" %0. Testler 85.
- **2026-10-05** — Veri seti + yüzde commit edildi (`45ceb64`). **Model eğitimi denendi** (`scripts/train_adapter.py`, `fashion_rec/adapter.py`): CLIP dondurulup görsel ve metin vektörlerine birer doğrusal katman (birim matrisle başlar, λ·‖W−I‖² cezası) eğitildi; karşıtlıklı (InfoNCE) kayıp, aynı metni taşıyan ürünler pozitif. Bölünme: eğitim 30.851 / doğrulama 4.407 / test 8.814 (test hiç görülmedi). Metin kaynağı × λ ızgarası doğrulamada:

  | metin kaynağı | etiket sorguları | Türkçe sorgular |
  |---|---|---|
  | CLIP (adaptörsüz) | 0,710 | **0,774** |
  | ürün adı, λ=0,1 | 0,688 | 0,747 |
  | **şablon ("Women Silver Watches"), λ=0,1** | **0,748** | 0,753 |
  | ad + şablon, λ=0,1 | 0,726 | 0,742 |

  Testte seçilen ayar: etiket 0,754 → 0,768, Türkçe 0,742 → 0,753, görsel renk 0,355 → 0,379, görsel tür 0,821 → 0,816. Ürün adlarıyla eğitim zarar verdi (marka/model kelimelerine uyum). **Karar: adaptör kullanılmıyor** — kazanç 1–2 puan, küçük sorgu setlerinin gürültü payı içinde ve doğrulamada Türkçeyi geriletiyor. Altyapı yerinde: adaptör dosyası varsa `ClipModel` otomatik uygular ve önbellek anahtarı adaptör imzasını içerir (4 test).
  Değerlendirme setinde hata bulundu ve düzeltildi: "parfüm" beklenen etiketi "Perfume and body mist" yazılmıştı (doğrusu "Body Mist"); "parfüm" aslında 1,0, metin araması ortalaması 0,74 değil **0,79**. **Asıl Türkçe darboğazı kelime dağarcığı**: "kemer" 0 / "belt" 1,0, "terlik" 0,3 / "flip flops" 1,0, "yeşil kurta" 0,3 / "green kurta" 0,6. Sıradaki adım önerisi: Türkçe moda terimleri sözlüğüyle sorguyu İngilizceye genişletme (eğitim gerektirmez).
- **2026-10-05** — Adaptör denemesi commit edildi (`ba6482b`). **Türkçe moda sözlüğü** eklendi (`fashion_rec/glossary.py`, ~200 terim): kök eşleme (ekli haller), önce çok kelimeli ifadeler, kısa kökler (≤3 harf) en fazla 3 harf ek alır ("bot" ≠ "botanik"), Türkçe büyük/küçük harf (I → ı, İ → i), boş karşılık kelimeyi atar ("rengi"). Önce **ayrık değerlendirme seti** hazırlandı (`notebooks/turkish_queries.py`, 30 sorgu, sözlük ayarlanırken kullanılmadı). `notebooks/compare_glossary.py` (tüm katalog, ilk 10 isabet):

  | | sözlüksüz | yalnız çeviri | özgün + çeviri ortalaması |
  |---|---|---|---|
  | geliştirme (19) | 0,805 | **0,884** | 0,879 |
  | ayrık (30) | 0,680 | **0,830** | 0,833 |

  Yöntem geliştirme setinde seçildi: **yalnız çeviri** (`text_search.encode_query`; metin araması ve "bunun gibi ama…" değişikliği). Kazananlar: "erkek kemer" 0 → 1,0, "sütyen" 0,1 → 0,9, "kazak" 0 → 0,9, "terlik" 0,3 → 1,0, "kadın tayt" 0,3 → 0,8. Gerileyenler: "topuklu ayakkabı" 1,0 → 0,7, "erkek şort" 1,0 → 0,7, "mor elbise" 0,5 → 0,3, "beyaz tişört" 0,8 → 0,7. Arayüz çeviriyi "Sözlükle aranan: …" diye gösteriyor. Testler 107 (sözlük + sahte modelle Türkçe sorgunun aramaya ulaştığı test). Model eğitimiyle (adaptör, +1 puan) karşılaştırınca sözlük çok daha etkili: Türkçe darboğazı kelime dağarcığıydı.
- **2026-10-06** — Sözlük commit edildi (`21ce333`); 44 bin ürünlük veri seti indirilmişse varsayılan oldu (`ef1c160`; testler `FASHION_DATASET=women` ile ortamdan bağımsız). **Bölüm bazında ayarlar** (kullanıcının ilk istediği Erkek / Kadın / Erkek Çocuk / Kız Çocuk / Aksesuar ayrımı): `catalog.segment_of` (aksesuar cinsiyetten önce), arayüzde "Aynı bölümden öner" (varsayılan açık), yüklenen görsel için 20 komşu çoğunluğuyla bölüm tahmini, bölüme göre varsayılan renk önemi, bölüm içi yüzde ölçeği (`calibration.image_scale/combined_scale(subset=…)`). `scripts/tune_segments.py` her bölüm için renk önemini 0–0,6 arasında tarıyor (hedef: tür + renk isabeti ortalaması; sorguların yarısıyla seçilip yarısında ölçülüyor) ve `fashion_rec/segment_settings.json`'a yazıyor. Seçilenler: Kadın 0,2, Erkek 0,3, Kız Çocuk 0,2, Erkek Çocuk 0,4, Aksesuar 0,5, Unisex 0,4. Aynı bölüm kısıtlaması cinsiyet isabetini çocuklarda 0,85 → 1,00, Kadın/Erkek'te 0,94–0,96 → 1,00 yapıyor; ürün türü de hafif artıyor.
  **Etiket hatası bulundu**: adında "girl" geçen 185 ürün "Women", "boy" geçen 210 ürün "Men" etiketli (çocuk ürünlerinin ~%23'ü; "Doodle Girl", "Palm Tree Kids Girls" gibi). `catalog.corrected_gender` adına göre düzeltiyor (kelime sınırı: "boyfriend" etkilenmez); bölümler, Cinsiyet filtresi ve ayar ölçümü düzeltilmiş cinsiyeti kullanıyor. Kız Çocuk 627 → 807, Erkek Çocuk 813 → 1.030 ürün. Not: hedef "tür ve renk eşit önemli" varsayımına dayanıyor; renk önemi arttıkça tür isabeti düşüyor (ör. Erkek Çocuk 0,921 → 0,864). Testler 112.
- **2026-10-06** — Bölüm ayarları commit edildi (`a603483`). **Çeşitlilik** eklendi. Ölçüm: bölüm içi ilk 5 öneride sorguların %28'inde aynı adlı ürün tekrarı, %23'ünde sorguyla aynı adlı ürün. Ama aynı adlı ürün çiftlerinin görsel benzerlik medyanı 0,85 (rastgele çift 0,45) ve yalnız %6'sı > 0,97: çoğu "Men Navy Blue T-shirt" gibi genel adlı farklı ürünler → ada göre eleme yanlış olurdu. Görsel **MMR** (`fashion_rec/diversity.py`): puan = (1−d)·ilgi − d·(seçilenlere en yüksek benzerlik); ilk 100 aday yeniden sıralanır, skorlar değişmez. İki hata ölçümle/testle bulundu: (1) ilgiyi 0–1'e ölçeklemek onu baskın kılıyordu (d = 0,5'te bile neredeyse değişiklik yok) → ilgi sorguyla cosine aralığına ölçeklendi; (2) sorguyu "seçilmiş" saymak ilgiyi kendi kendine götürüyordu (test: alakasız ürün ilgili ürünü yendi) → sorgu yalnız neredeyse kopyalar (> 0,97) için ceza. `notebooks/compare_diversity.py` (1.000 sorgu): d = 0 tür 0,876 / kopya %14; **d = 0,2 tür 0,866 / kopya %9** (seçilen: ≤ 1 puan kayıp); d = 0,4 0,838 / %8; d = 0,6 0,772 / %0. Arayüzde "Çeşitlilik" kaydırıcısı (0–0,6, varsayılan 0,2), filtrelerden sonra üç modda da uygulanıyor. Örnek: Gini and Jony kız çocuk üstü → önceden mor, mor, kırmızı, kırmızı; şimdi mor, pembe, beyaz, çizgili pembe. Testler 116.
- **2026-10-06** — Çeşitlilik commit edildi (`136ba8b`). **Kombin önerisi** eklendi (`fashion_rec/outfit.py`, arayüzde "Kombin önerisi" sekmesi). Veri setinde kombin etiketi yok → kurallara dayalı: rol (alt kategoriden) → tamamlayıcı yuvalar; aynı cinsiyet + uyumlu kullanım (Günlük / Resmi / Spor / Etnik; veri setinde 34 bin / 2,3 bin / 4 bin / 3,2 bin); sıralama = CLIP stil (yuva içi z) + kullanım birebir + renk uyumu + mevsim; yuva başına 3 parça, aynı modelin renkleri MMR ile, neredeyse aynı görseller (> 0,97) kesin elenerek.
  Gözle kontrolde bulunan ve düzeltilen sorunlar: (1) adıyla etiketi çelişen ürünler ("Wrangler Men Rider Solid White Shirts" terlik etiketli; görseli gömlek) → ad kontrolü (`name_conflicts`, yuvaları etkileyen ~20 ürün); (2) görseli adıyla uyuşmayan ürünler ("… Black Wallet" görselinde ekose gömlek) stil benzerliği yüzünden öne çıkıyordu → aday görselinin CLIP'e göre yuvanın rolüne benzemesi şart (`clip_roles`, önbellek `visual_roles.npz`; CLIP rol tahmini %94,6 isabetli). CLIP doğrulamasını genel etiket düzeltmesi olarak kullanmak denendi ve bırakıldı: uyuşmazlıkların çoğu CLIP'in hatasıydı (üstle giyilen kot "üst"). (3) Kot "Blue" etiketli olduğu için canlı renk sayılıp kırmızı üste ceza alıyordu → kot nötr ("Denim").
  Vekil ölçütler (500 sorgu): kullanım birebir rastgele 0,84 / yalnız CLIP 0,89 / sezgisel 0,97; renk uyumu 0,88 / 0,81 / 0,98 — sezgisel bu kuralları kullandığından yalnızca kuralların işlediğini gösterir. Sınırlama: desen uyumu yok. Testler 123.
- **2026-10-06** — Kombin önerisi commit edildi (`085d86e`). **Kullanıcı testinden düzeltme** ("Doodle Boy Athlethic Red Tshirts"): (1) seçilen ürünün başka bir ilanı %95,8 ile öneriliyordu — çeşitlilik sorgunun kopyalarını yalnızca cezalandırıyordu → kopyalar (benzerlik > 0,97; sorguya ya da başka bir öneriye) artık tamamen çıkarılıyor (`mmr(drop_near_duplicates=True)` sorgu kopyalarını da eler); (2) kartlar MMR seçim sırasıyla gösteriliyordu (1. %66,6, 2. %95,8) → seçilenler benzerliğe göre azalan sıralanıyor; (3) yüzde referansı örnek aramaların ilk 10'undaki kopya ilanları da içeriyordu (≈ %100 benzerlik), çıtayı yükseltip farklı ürünleri düşük gösteriyordu → referans da kopyalar elenerek hesaplanıyor (`calibration._dedupe`). Aynı örnekte öneriler %54–67 yerine %71–82. Testler 124.
- **2026-10-07** — Kopya/yüzde düzeltmesi commit edildi (`e6aa4d7`). **H&M veri seti eklendi** (öğrenme amaçlı; haklar H&M'e ait, Kaggle yarışma kuralları). Sitesinden toplama (scraping) kullanım koşulları ve telif nedeniyle yapılmadı. Etiketler HF `Qdrant/hm_ecommerce_products` (105 bin ürün; görsel bağlantıları ölü — S3 deposu kaldırılmış), görseller HF `microsoft/hnm-search-data` deposundan tek tek (orijinal 1166×1750). `scripts/import_hm.py`: bölümlere dengeli 25 bin ürün (Kadın %32, Erkek %24, Kız Çocuk %14, Erkek Çocuk %12, Aksesuar %18), ürün başına en fazla 2 renk çeşidi, görseller 512 piksel yüksekliğe küçültülüyor (~25 KB); etiketler `metadata.csv` biçimine eşleniyor (Ladieswear / Divided → Women, Menswear → Men, çocuk bölümlerinde "Girl" / "Boy", cinsiyeti belirsiz çocuk ürünleri alınmıyor; kullanım yalnız Spor / Günlük; kot adından ya da denim grubundan; "Khaki green" → Olive (nötr)). Sonuç: 24.877 ürün, 607 MB, ~70 dk (saniyede ~6 görsel). CLIP 124 sn. `FASHION_DATASET=hm`.
  Bölüm ayarları (`tune_segments.py`): aynı bölüm kısıtlaması H&M'de daha önemli — cinsiyet isabeti tüm katalogda Kadın 0,80, Kız Çocuk 0,63, Erkek Çocuk 0,69 → hepsi 1,00. Ürün türü isabeti düşük görünüyor (Kadın 0,64) çünkü H&M türleri çok ince ("Vest top" / "Top" / "T-shirt" ayrı). Seçilen renk önemi: Kadın / Erkek 0,1, Kız / Erkek Çocuk / Aksesuar 0,3. Tarayıcıda: erkek çocuk kumaş pantolonu → benzer: siyah pantolonlar; kombin: yelek, blazer, kasket. Bilinen eksikler: çocuk ayakkabıları cinsiyeti belirsiz bölümde olduğu için alınmadı (çocuk kombininde ayakkabı yuvası yok); metin araması cinsiyet bilmiyor (H&M adlarında "girl" yok). Ürün seçici araması artık kelime başına bakıyor ("dress" ≠ "welldressed").
- **2026-10-07** — H&M commit edildi (`db420f2`), ham indirmeler silindi. **Satın alma verisi** (H&M, HF `microsoft/hnm-search-data` → transactions_train.parquet, 774 MB): katalog ürünlerini içeren 5,6 milyon satır; sepet = müşteri + gün (en fazla 30 ürün); eş-satın alma matrisi seyrek çarpımla (`scripts/build_copurchase.py`, ~2 dk). **Kombin önerisinin ilk gerçek ölçümü** (`notebooks/evaluate_copurchase.py`): 2020-06-22 öncesi 872 bin sepetle öğrenme, sonraki 3 ayın 124 bin sepetinden 4.000 tamamlayıcı çift; yuvadaki ~2.200 aday arasında gerçekten alınan parçanın sırası. İlk 10: rastgele 0,004, **kurallar 0,026**, çok satanlar 0,043, birlikte alınma 0,087, karma 0,088. Geçmişi olan ürünlerde (çiftlerin %71'i) birlikte alınma 0,109 / kurallar 0,022 (~5 kat); yeni ürünlerde hepsi ~0,035 ama kurallar hedefi çok daha yukarı koyuyor (medyan sıra 444'e karşı 1.135). Değerlendirmede bir yanlılık bulundu ve düzeltildi: eşit puanlılar hedefin lehine sayılıyordu (yeni ürünlerin hepsi popülerlik 0 alınca çok satanlar / birlikte alınma olduğundan iyi görünüyordu) → eşitlikler yarı yarıya sayılıyor. Ders: gözle makul görünen kurallar gerçek alışverişi çok satanlardan bile kötü tahmin ediyor.
  Uygulama: `fashion_rec/copurchase.py` (`PURCHASE_WEIGHTS` = birlikte 8, çok satanlar 1; MRR ile seçildi), `OutfitCatalog` birlikte alınma / popülerlik / stil ağırlıkları alıyor (`candidates` + `scores` ayrıldı), H&M'de üçüncü sekme "Birlikte alınanlar" (aynı ürünün renkleri hariç, sepet sayısıyla). Tam matris: 996 bin sepet, 1,59 milyon çift, 6,9 MB. Testler 127.
- **2026-10-07** — Satın alma verisi commit edildi (`860858b`), ham kayıtlar silindi. **Hızlı iyileştirmeler**: (1) **metin aramasında cinsiyet** — `glossary.gender_in_query` ("kız çocuk elbisesi" → Kız Çocuk; çocuk ifadeleri önce; İngilizce kelimeler tam eşleşir, "man" ≠ "manto"), sonuçlar ürünün cinsiyetine göre süzülür (bölüme göre süzmek aksesuarları elerdi); H&M'de "kız çocuk elbisesi" artık yalnız kız çocuk elbiseleri. (2) **Veri seti seçici** — `config.set_dataset()`; modüller veri setine bağlı değerleri içe aktararak sabitlemiyor, çağrı anında `config.X` okuyor (9 modül); arayüz değişimde yalnız veri setine bağlı önbellekleri temizliyor (CLIP / VGG16 modelleri kalıyor); tek sunucu yeterli. (3) **H&M çocuk ayakkabıları** — H&M bunları cinsiyetsiz tutuyor → "Kids" / "Çocuk" cinsiyeti; `catalog.compatible_segments / compatible_genders`: Çocuk ürünleri kız ve erkek çocuğa uyar, yetişkine gelmez. İçe aktarma sabit tohumla yeniden çalıştırıldı: eski 24.877 ürün aynen kaldı, 1.220 çocuk ayakkabısı eklendi (26.097); önbellekler, bölüm ayarları ve birlikte alınma matrisi yeniden hesaplandı (1.001.769 sepet). Kız çocuk elbisesinin kombininde artık çocuk ayakkabıları var. Eksik: çocuk aksesuarlarının çoğu da cinsiyetsiz bölümde, alınmadı (çocuk kombinlerinde aksesuar az). Değerlendirme dosyaları (`copurchase_train`, `test_baskets`) eski katalog boyutunda kaldığı için silindi; `--cutoff` ile yeniden üretilebilir. Testler 140.
