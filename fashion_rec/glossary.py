"""Türkçe moda terimleri sözlüğü: sorgudaki bilinen terimleri İngilizceye çevirir.

Çok dilli CLIP bazı Türkçe kelimeleri tanımıyor: "kemer" araması hiç kemer
getirmezken "belt" hepsini getiriyor. Sözlük, sorgudaki bilinen terimlerin
İngilizce karşılığını çıkarır; arama bu karşılıkla da yapılır.

Eşleme kökler üzerinden yapılır, böylece ekli halleri de tanınır ("gömleği",
"çantası", "elbisesi"). Çok kelimeli ifadeler ("kol saati", "spor ayakkabı")
tek kelimelerden önce denenir. Boş karşılık ("") kelimeyi atar ("rengi").
"""

import re

# (Türkçe kök[ler], İngilizce karşılık); çok kelimeli ifadeler boşlukla ayrılır
_ENTRIES = [
    # çok kelimeli ifadeler
    ("dizüstü bilgisayar çanta", "laptop bag"), ("laptop çanta", "laptop bag"), ("sırt çanta", "backpack"),
    ("el çanta", "handbag"), ("spor çanta", "duffel bag"), ("bel çanta", "waist pouch"),
    ("kol saat", "wrist watch"), ("güneş gözlü", "sunglasses"), ("spor ayakkabı", "sports shoes"),
    ("topuklu ayakkabı", "heels"), ("kot pantolon", "jeans"), ("kot ceket", "denim jacket"),
    ("takım elbise", "suit"), ("eşofman alt", "track pants"), ("eşofman üst", "sweatshirt"),
    ("kız çocuk", "girls"), ("erkek çocuk", "boys"), ("uzun kollu", "long sleeve"), ("kısa kollu", "short sleeve"),
    ("v yaka", "v-neck"), ("bisiklet yaka", "round neck"), ("kol düğme", "cufflinks"), ("iç çamaşır", "innerwear"),
    ("ten rengi", "nude"), ("gece elbise", "evening dress"),
    # giyim
    ("elbise", "dress"), ("abiye", "evening gown"), ("gömle", "shirt"), ("tişört", "t-shirt"), ("tisort", "t-shirt"),
    ("pantolon", "trousers"), ("kot", "jeans"), ("şort", "shorts"), ("bermuda", "shorts"), ("etek", "skirt"),
    ("bluz", "blouse top"), ("tunik", "tunic"), ("kazak", "sweater"), ("hırka", "cardigan sweater"),
    ("ceket", "jacket"), ("mont", "jacket"), ("yağmurluk", "rain jacket"), ("kaban", "coat"), ("palto", "coat"),
    ("trençkot", "trench coat"), ("yelek", "waistcoat"), ("eşofman", "tracksuit"), ("tayt", "leggings"),
    ("tulum", "jumpsuit"), ("salopet", "dungarees"), ("pijama", "pyjamas night suit"), ("gecelik", "nightdress"),
    ("sabahlık", "bath robe"), ("bornoz", "bath robe"), ("mayo", "swimwear"), ("bikini", "bikini swimwear"),
    ("sütyen", "bra"), ("külot", "briefs"), ("boxer", "boxers"), ("atlet", "vest"), ("takım", "set"),
    # etnik
    ("sari", "saree"), ("kurta", "kurta"), ("kurti", "kurti"), ("şalvar", "salwar"), ("lehenga", "lehenga"),
    # ayakkabı
    ("ayakkabı", "shoes"), ("topuklu", "heels"), ("terlik", "flip flops slippers"), ("sandalet", "sandals"),
    ("babet", "flats"), ("çizme", "boots"), ("bot", "boots"), ("spor", "sports"),
    # aksesuar
    ("çanta", "bag"), ("cüzdan", "wallet"), ("kemer", "belt"), ("saat", "watch"), ("gözlü", "glasses"),
    ("küpe", "earrings"), ("kolye", "necklace"), ("yüzük", "ring"), ("bileklik", "bracelet"), ("bilezik", "bangle bracelet"),
    ("şapka", "hat cap"), ("bere", "beanie cap"), ("atkı", "scarf muffler"), ("eşarp", "scarf"), ("fular", "scarf"),
    ("şal", "stole shawl"), ("kravat", "tie"), ("çorap", "socks"), ("eldiven", "gloves"), ("şemsiye", "umbrella"),
    ("anahtarlık", "key chain"), ("takı", "jewellery"),
    # kozmetik
    ("ruj", "lipstick"), ("oje", "nail polish"), ("parfüm", "perfume"), ("deodorant", "deodorant"),
    ("rimel", "mascara"), ("maskara", "mascara"), ("sürme", "kajal eyeliner"), ("fondöten", "foundation"),
    ("allık", "blush"), ("göz farı", "eyeshadow"),
    # cinsiyet / yaş
    ("kadın", "women"), ("bayan", "women"), ("erkek", "men"), ("çocuk", "kids"), ("bebek", "baby"),
    # renkler
    ("siyah", "black"), ("beyaz", "white"), ("kırmızı", "red"), ("lacivert", "navy blue"), ("mavi", "blue"),
    ("yeşil", "green"), ("sarı", "yellow"), ("mor", "purple"), ("pembe", "pink"), ("gri", "grey"),
    ("kahverengi", "brown"), ("kahve", "brown"), ("bej", "beige"), ("krem", "cream"), ("turuncu", "orange"),
    ("bordo", "maroon"), ("altın", "gold"), ("gümüş", "silver"), ("haki", "khaki olive"), ("lila", "lavender"),
    ("eflatun", "lavender"), ("turkuaz", "turquoise"), ("füme", "charcoal"), ("renkli", "multicolour"),
    ("rengi", ""), ("renk", ""), ("açık", "light"), ("koyu", "dark"),
    # desen / kumaş / kesim
    ("çizgili", "striped"), ("kareli", "checked"), ("ekose", "checked"), ("çiçekli", "floral"),
    ("desenli", "printed"), ("baskılı", "printed"), ("puantiyeli", "polka dot"), ("düz", "solid"),
    ("payetli", "sequined"), ("simli", "shimmer"), ("dantel", "lace"), ("deri", "leather"), ("kadife", "velvet"),
    ("saten", "satin"), ("ipek", "silk"), ("yün", "wool"), ("pamuk", "cotton"), ("keten", "linen"),
    ("kolsuz", "sleeveless"), ("kapüşonlu", "hooded"), ("yakalı", "collared"),
    # kullanım
    ("resmi", "formal"), ("günlük", "casual"), ("parti", "party"), ("düğün", "wedding"), ("gece", "night"),
]

# kısa kökler yanlış eşleşmesin diye ("bot" → "botanik") en fazla bu kadar ek alır
_SHORT_STEM, _SHORT_SUFFIX = 3, 3

_PHRASES = sorted(((tuple(tr.split()), en) for tr, en in _ENTRIES), key=lambda e: -len(e[0]))


def turkish_lower(text: str) -> str:
    return text.replace("I", "ı").replace("İ", "i").lower()


def _matches(token: str, stem: str) -> bool:
    if not token.startswith(stem):
        return False
    return len(stem) > _SHORT_STEM or len(token) <= len(stem) + _SHORT_SUFFIX


def translate(text: str) -> str | None:
    """Sorgunun İngilizce karşılığı; sözlükte hiç terim bulunmazsa None.

    Sözlükte olmayan kelimeler olduğu gibi kalır (ör. marka adları).
    """
    tokens = re.findall(r"[\wçğıöşüâîû-]+", turkish_lower(text))
    out, found, i = [], False, 0
    while i < len(tokens):
        for stems, english in _PHRASES:
            n = len(stems)
            if all(i + k < len(tokens) and _matches(tokens[i + k], stems[k]) for k in range(n)):
                if english:
                    out.append(english)
                found, i = True, i + n
                break
        else:
            out.append(tokens[i])
            i += 1
    return " ".join(out) if found else None
