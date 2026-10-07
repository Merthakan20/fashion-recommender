"""Kombin önerisi: seçilen ürünü tamamlayan parçalar (ör. gömleğe uyan pantolon, ayakkabı, aksesuar).

Veri setinde "birlikte giyilir" bilgisi olmadığından bu öğrenilmiş bir model
değil, etiketlere dayalı bir sezgiseldir:
1. Ürünün rolü alt kategorisinden çıkar (üst / alt / tek parça / ayakkabı /
   aksesuar) ve role göre tamamlayıcı yuvalar seçilir.
2. Adaylar aynı cinsiyetten (veya unisex) ve uyumlu kullanımdandır
   (resmi ↔ resmi / yarı resmi, spor ↔ spor / günlük, …).
3. Yuvadaki adaylar şu toplamla sıralanır: CLIP görsel stil benzerliği
   (yuva içinde z-skoru) + kullanım birebir eşleşmesi + renk uyumu + sezon.
"""

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .catalog import Metadata, corrected_gender
from .diversity import mmr
from .features import cached_features

ROLES = {
    "üst": {"Topwear"},
    "alt": {"Bottomwear"},
    "tek parça": {"Dress", "Saree", "Apparel Set"},
    "ayakkabı": {"Shoes", "Sandal", "Flip Flops"},
    "aksesuar": {"Watches", "Bags", "Belts", "Eyewear", "Jewellery", "Wallets", "Headwear", "Ties", "Scarves",
                 "Stoles", "Cufflinks", "Mufflers"},
}
ROLE_OF = {sub: role for role, subs in ROLES.items() for sub in subs}
# ürün adındaki rol kelimeleri: etiket hatalarını yakalamak için (ör. "… White Shirts" adlı ürün terlik etiketli)
_ROLE_WORDS = {
    "üst": r"shirts?|t-?shirts?|tshirts?|tops?|kurtas?|kurtis?|tunics?|sweaters?|sweatshirts?|jackets?|blazers?",
    "alt": r"jeans|trousers?|shorts|skirts?|leggings|track ?pants?|capris|jeggings",
    "tek parça": r"dress(es)?|sarees?|frock",
    "ayakkabı": r"shoes?|sandals?|flip ?flops?|heels|flats|sneakers|slippers?|boots?|floaters?",
    "aksesuar": r"watch(es)?|bags?|backpacks?|belts?|sunglasses|wallets?|earrings?|necklace|pendant|ring|bracelet"
                r"|caps?|scarf|scarves|tie|stole|cufflinks|clutch(es)?|handbags?",
}
_ROLE_PATTERNS = {role: re.compile(rf"\b({words})\b", re.IGNORECASE) for role, words in _ROLE_WORDS.items()}


def name_conflicts(row: dict[str, str]) -> bool:
    """Ürün adı başka bir rolü söylüyor, kendi rolünü hiç söylemiyorsa True (etiket hatası şüphesi).

    CLIP ile görsel doğrulama denendi ama işaretlenenlerin çoğu CLIP'in kendi
    hatasıydı (üstle giyilen kot "üst", atkı "üst" sanılıyor); ad kontrolü
    yalnız bariz hataları yakalar (~20 ürün: üst etiketli kemerler, terlik
    etiketli gömlek, alt etiketli yüzücü boneleri).
    """
    role = role_of(row)
    hits = {r for r, pattern in _ROLE_PATTERNS.items() if pattern.search(row.get("name", ""))}
    return bool(role and hits and role not in hits)


# CLIP ile "görsel bu role benziyor mu" kontrolü için tarifler
ROLE_PROMPTS = {
    "üst": "a photo of a top, shirt, t-shirt, kurta or jacket",
    "alt": "a photo of trousers, jeans, shorts, a skirt or leggings",
    "tek parça": "a photo of a dress or a saree",
    "ayakkabı": "a photo of shoes, sandals or flip flops",
    "aksesuar": "a photo of an accessory such as a watch, bag, belt, sunglasses, jewellery or wallet",
}


def clip_roles(paths: list[Path], clip_features: np.ndarray, cache_path: Path, model=None) -> np.ndarray:
    """Her görselin CLIP'e göre en çok benzediği rol (önbellekli; geçerliyse model yüklenmez).

    Kombin yuvalarında adayın görseli yuvanın rolüne benzemeli: bazı ürünlerin
    görseli adıyla uyuşmuyor (ör. "… Black Wallet" görselinde ekose gömlek) ve
    stil benzerliği bu görselleri sorguya benzedikleri için öne çıkarıyordu.
    """
    from .text_search import clip_key, load_clip_model, normalize

    def compute():
        clip = model or load_clip_model()
        text = normalize(clip.encode_text(list(ROLE_PROMPTS.values())))
        return np.argmax(clip_features @ text.T, axis=1)

    key = clip_key() + "/roles/" + "|".join(ROLE_PROMPTS.values())
    return np.array(list(ROLE_PROMPTS))[cached_features(paths, cache_path, compute, key=key)]


SLOTS = {  # rol -> kombinde tamamlayıcı yuvalar
    "üst": ["alt", "ayakkabı", "aksesuar"],
    "alt": ["üst", "ayakkabı", "aksesuar"],
    "tek parça": ["ayakkabı", "aksesuar"],
    "ayakkabı": ["üst", "alt", "aksesuar"],
    "aksesuar": ["üst", "alt", "ayakkabı"],
}
USAGE_COMPATIBLE = {
    "Casual": {"Casual", "Smart Casual", "Travel", "Party"},
    "Formal": {"Formal", "Smart Casual"},
    "Smart Casual": {"Smart Casual", "Casual", "Formal"},
    "Sports": {"Sports", "Casual"},
    "Ethnic": {"Ethnic", "Casual"},
    "Party": {"Party", "Casual"},
    "Travel": {"Travel", "Casual"},
}
NEUTRALS = {"Black", "White", "Grey", "Grey Melange", "Navy Blue", "Beige", "Brown", "Charcoal", "Cream",
            "Off White", "Khaki", "Tan", "Taupe", "Nude", "Skin", "Silver", "Steel", "Gold", "Coffee Brown",
            "Mushroom Brown", "Metallic", "Bronze", "Copper", "Denim", "Olive"}  # Olive: H&M "Khaki green"
DENIM_TYPES = {"Jeans", "Jeggings"}  # kot, renk etiketi "Blue" olsa da nötr sayılır
SEASON_GROUP = {"Summer": "sıcak", "Spring": "sıcak", "Fall": "soğuk", "Winter": "soğuk"}


@dataclass
class Weights:
    usage: float = 1.0    # kullanım birebir aynıysa
    colour: float = 0.5   # renk uyumu puanı (-1 … +1) ile çarpılır
    season: float = 0.3   # aynı mevsim grubundaysa


def role_of(row: dict[str, str]) -> str | None:
    return ROLE_OF.get(row.get("subCategory", ""))


def outfit_colour(row: dict[str, str]) -> str:
    """Renk uyumu için renk: kot ürünler "Denim" (nötr)."""
    return "Denim" if row.get("articleType") in DENIM_TYPES else row.get("baseColour", "")


def colour_harmony(a: str, b: str) -> float:
    """Basit renk kuralı: nötr renk her şeyle uyar (+1), aynı canlı renk tekdüze (-0,5), iki farklı canlı renk (-1)."""
    if not a or not b or a in NEUTRALS or b in NEUTRALS:
        return 1.0
    return -0.5 if a == b else -1.0


def gender_ok(query_gender: str, candidate_gender: str) -> bool:
    if candidate_gender == "Unisex" or query_gender == candidate_gender:
        return True
    return query_gender == "Unisex" and candidate_gender in {"Men", "Women"}


def zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std() if len(x) > 1 and x.std() > 0 else np.zeros_like(x)


class OutfitCatalog:
    """Katalog etiketlerinin dizi hali; kombin önerisinde 44 bin ürünü her seferinde baştan okumamak için."""

    def __init__(self, paths: list[Path], features: np.ndarray, metadata: Metadata,
                 visual_roles: np.ndarray | None = None):
        self.paths, self.features = paths, features
        rows = [metadata.get(p.name, {}) for p in paths]
        # adıyla çelişen (etiketi şüpheli) ürünler hiçbir yuvaya aday olmaz
        role = np.array(["" if name_conflicts(r) else role_of(r) or "" for r in rows])
        if visual_roles is not None:  # görseli etiketlenen role benzemeyenler de (bkz. clip_roles)
            role = np.where(visual_roles == role, role, "")
        self.role = role
        self.gender = np.array([corrected_gender(r) for r in rows])
        self.sub = np.array([r.get("subCategory", "") for r in rows])
        self.usage = np.array([r.get("usage", "") for r in rows])
        self.colour = np.array([outfit_colour(r) for r in rows])
        self.season = np.array([SEASON_GROUP.get(r.get("season", ""), "") for r in rows])

    def suggest(self, query_row: dict[str, str], query_vector: np.ndarray, per_slot: int = 3,
                weights: Weights = Weights(), exclude: Path | None = None) -> dict[str, list[tuple[Path, float]]]:
        """Yuva adı -> en iyi `per_slot` parça (yol, puan). Rolü olmayan ürün için boş sözlük.

        `query_row` ürünün etiketleri (katalog dışı görselde komşulardan tahmin,
        bkz. `infer_row`); `query_vector` görselin CLIP vektörü.
        """
        role = role_of(query_row)
        if role is None:
            return {}
        gender = corrected_gender(query_row)
        usage = query_row.get("usage", "") or "Casual"
        compatible = list(USAGE_COMPATIBLE.get(usage, {usage}))
        season = SEASON_GROUP.get(query_row.get("season", ""), "")
        colour = outfit_colour(query_row)
        if gender == "Unisex":
            gender_mask = np.isin(self.gender, ["Men", "Women", "Unisex"])
        else:
            gender_mask = (self.gender == gender) | (self.gender == "Unisex")
        not_query = np.array([p != exclude for p in self.paths]) if exclude is not None else True

        outfit = {}
        for slot in SLOTS[role]:
            mask = (self.role == slot) & gender_mask & not_query
            if slot == "aksesuar":  # aynı tür aksesuarı önerme (saate saat)
                mask &= self.sub != query_row.get("subCategory", "")
            matching = mask & np.isin(self.usage, compatible)
            idx = np.flatnonzero(matching if matching.any() else mask)  # uyumlu aday yoksa kullanım şartı gevşer
            if not len(idx):
                continue
            score = zscore(self.features[idx] @ query_vector)
            score += weights.usage * (self.usage[idx] == usage)
            score += weights.colour * np.array([colour_harmony(colour, c) for c in self.colour[idx]])
            if season:
                score += weights.season * (self.season[idx] == season)
            order = np.argsort(-score)
            ranked = [(self.paths[i], float(score[j])) for j, i in zip(order, idx[order])]
            # aynı modelin renkleri yuvayı doldurmasın
            outfit[slot] = mmr(ranked, self.paths, self.features, per_slot, diversity=0.3, drop_near_duplicates=True)
        return outfit


def infer_row(neighbours: list[tuple[Path, float]], metadata: Metadata, k: int = 20) -> dict[str, str]:
    """Katalog dışı görselin etiketleri: en yakın `k` komşuda her alanın en sık değeri."""
    from collections import Counter

    row = {}
    for field in ("subCategory", "gender", "usage", "season", "baseColour", "masterCategory"):
        values = Counter(metadata.get(p.name, {}).get(field, "") for p, _ in neighbours[:k])
        values.pop("", None)
        if values:
            row[field] = values.most_common(1)[0][0]
    return row
