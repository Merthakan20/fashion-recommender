"""Önerileri çeşitlendirme: neredeyse aynı ürünler sonuçları doldurmasın.

44 bin üründe aynı ürünün farklı ilanları ve aynı modelin renkleri ilk
sıraları kaplayabiliyor. MMR (Maximal Marginal Relevance) yeniden sıralama:
her adımda, sorguya benzerliği yüksek ama zaten seçilmiş önerilere (ve
sorgunun kendisine) benzerliği düşük aday seçilir:

    puan = (1 - d) · ilgi − d · en yüksek benzerlik(aday, seçilenler)

Sorgunun kendisi yalnızca neredeyse kopyaları (benzerlik > NEAR_DUPLICATE,
ör. aynı ürünün başka ilanı) için tekrar sayılır; aksi halde sorguya
benzeyen her aday ilgisi kadar ceza alır ve ilgi kendi kendini götürür.

`d` = çeşitlilik ağırlığı (0 = sıralama değişmez). İlgi, adayların
skorlarının sorguyla görsel benzerlik (cosine) aralığına ölçeklenmiş
halidir: sıra korunur (z-skoru toplamı gibi skorlarla da çalışır) ama ilgi
ve tekrar terimi aynı ölçekte olur. 0–1'e ölçeklemek ilgiyi baskın kılıyor,
çünkü adayların birbirine benzerliği yalnızca ~0,85–0,95 arasında değişiyor.
"""

from pathlib import Path

import numpy as np

from .data import catalog_index

NEAR_DUPLICATE = 0.97  # bu benzerliğin üstü "aynı ürün" sayılır


def mmr(results: list[tuple[Path, float]], paths: list[Path], features: np.ndarray, top_n: int,
        diversity: float = 0.0, query_vector: np.ndarray | None = None,
        pool: int = 100, drop_near_duplicates: bool = False) -> list[tuple[Path, float]]:
    """`results` (azalan skorlu) içinden ilk `pool` adayı MMR ile yeniden sıralayıp `top_n` döndürür.

    Skorlar değişmez, yalnızca sıra değişir. `diversity` 0 ise (ve kopya
    eleme kapalıysa) ilk `top_n` olduğu gibi döner. `drop_near_duplicates`
    seçilmiş bir öneriyle neredeyse aynı (> NEAR_DUPLICATE) adayları tamamen
    eler; `query_vector` verilmişse sorgunun kopyaları da elenir (ör. seçilen
    ürünün başka bir ilanı önerilmesin, aynı ürünün iki ilanı yan yana çıkmasın).
    """
    if not 0 <= diversity <= 1:
        raise ValueError("diversity 0 ile 1 arasında olmalı")
    if (diversity == 0 and not drop_near_duplicates) or len(results) <= 1:
        return results[:top_n]

    candidates = results[:max(pool, top_n)]
    index = catalog_index(paths)[1]
    rows = [index[p.resolve()] if p.resolve() in index else None for p, _ in candidates]
    keep = [i for i, r in enumerate(rows) if r is not None]
    candidates = [candidates[i] for i in keep]
    vectors = features[[rows[i] for i in keep]]

    scores = np.array([s for _, s in candidates], dtype=float)
    span = scores.max() - scores.min()
    relevance = (scores - scores.min()) / span if span > 0 else np.ones_like(scores)
    if query_vector is not None:
        to_query = vectors @ query_vector
        relevance = to_query.min() + relevance * (to_query.max() - to_query.min())

    # her adayın şimdiye kadar seçilenlere en yüksek benzerliği; sorgu yalnız neredeyse kopyalar için
    redundancy = np.full(len(candidates), -1.0)
    if query_vector is not None:
        redundancy = np.where(to_query > NEAR_DUPLICATE, to_query, -1.0)
    chosen, available = [], np.ones(len(candidates), dtype=bool)
    if drop_near_duplicates and query_vector is not None:
        available &= to_query <= NEAR_DUPLICATE
    while len(chosen) < min(top_n, len(candidates)):
        value = np.where(available, (1 - diversity) * relevance - diversity * redundancy, -np.inf)
        best = int(np.argmax(value))
        if value[best] == -np.inf:  # kalan adayların hepsi elendi
            break
        chosen.append(best)
        available[best] = False
        to_best = vectors @ vectors[best]
        if drop_near_duplicates:
            available &= to_best <= NEAR_DUPLICATE
        redundancy = np.maximum(redundancy, to_best)
    return [candidates[i] for i in chosen]
