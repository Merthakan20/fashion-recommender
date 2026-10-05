"""Komut satırı arayüzü.

    python -m fashion_rec index [--force] [--text]
    python -m fashion_rec recommend <görsel> [-n 5] [-c 0.4]
    python -m fashion_rec search "<metin>" [-n 5]
    python -m fashion_rec combine <görsel> "<metin>" [-n 5] [-w 0.5]
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # TensorFlow log kalabalığını gizle

from .color import build_color_index  # noqa: E402
from .features import build_index  # noqa: E402
from .recommend import recommend  # noqa: E402
from .text_search import build_text_index, combined_search, load_clip_model, search  # noqa: E402


def positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("1 veya daha büyük olmalı")
    return n


def weight(value: str) -> float:
    w = float(value)
    if not 0 <= w <= 1:
        raise argparse.ArgumentTypeError("0 ile 1 arasında olmalı")
    return w


def cmd_index(args) -> None:
    paths, features = build_index(force=args.force)
    print(f"{len(paths)} görsel indekslendi, vektör boyutu: {features.shape[1]}")
    if args.text:
        paths, features = build_text_index(force=args.force)
        print(f"{len(paths)} görsel metin araması için indekslendi, vektör boyutu: {features.shape[1]}")


def cmd_recommend(args) -> None:
    image = Path(args.image)
    if not image.is_file():
        sys.exit(f"Hata: görsel bulunamadı: {image}")

    paths, features = build_index()
    color_features = build_color_index()[1] if args.c > 0 else None
    results = recommend(image, paths, features, top_n=args.n,
                        color_features=color_features, color_weight=args.c)

    print(f"Girdi: {image.name}" + (f" (renk önemi {args.c})" if args.c > 0 else "") + "\n")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {score:+.2f}  {path.name}" if args.c > 0 else f"{rank:>2}. {score:.3f}  {path.name}")


def cmd_search(args) -> None:
    query = args.query.strip()
    if not query:
        sys.exit("Hata: arama metni boş olamaz")

    model = load_clip_model()
    paths, features = build_text_index(model=model)
    results = search(query, paths, features, top_n=args.n, model=model)

    print(f"Sorgu: {query}\n")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {score:.3f}  {path.name}")


def cmd_combine(args) -> None:
    image = Path(args.image)
    if not image.is_file():
        sys.exit(f"Hata: görsel bulunamadı: {image}")
    text = args.text.strip()
    if not text:
        sys.exit("Hata: değişiklik metni boş olamaz")

    model = load_clip_model()
    paths, features = build_text_index(model=model)
    results = combined_search(image, text, paths, features, top_n=args.n, text_weight=args.w, model=model)

    print(f"Girdi: {image.name} + \"{text}\" (metin ağırlığı {args.w})\n")
    for rank, (path, score) in enumerate(results, start=1):
        print(f"{rank:>2}. {score:+.2f}  {path.name}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="fashion_rec", description="Görsel benzerliğine dayalı kıyafet önerisi")
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="özellikleri çıkar ve önbellekle")
    p_index.add_argument("--force", action="store_true", help="önbelleği yok sayıp yeniden hesapla")
    p_index.add_argument("--text", action="store_true", help="metin araması (CLIP) indeksini de oluştur")
    p_index.set_defaults(func=cmd_index)

    p_rec = sub.add_parser("recommend", help="bir görsele benzeyen kıyafetleri öner")
    p_rec.add_argument("image", help="girdi görselinin yolu")
    p_rec.add_argument("-n", type=positive_int, default=5, help="öneri sayısı (varsayılan: 5)")
    p_rec.add_argument("-c", type=weight, default=0.0,
                       help="renk önemi: 0 = yalnız VGG16, 1 = yalnız renk (varsayılan: 0)")
    p_rec.set_defaults(func=cmd_recommend)

    p_search = sub.add_parser("search", help="metne uyan kıyafetleri bul (CLIP, Türkçe veya İngilizce)")
    p_search.add_argument("query", help='arama metni, ör. "kırmızı payetli elbise"')
    p_search.add_argument("-n", type=positive_int, default=5, help="sonuç sayısı (varsayılan: 5)")
    p_search.set_defaults(func=cmd_search)

    p_comb = sub.add_parser("combine", help='görsele benzeyen ama metne uyan kıyafetler ("bunun gibi ama kırmızı")')
    p_comb.add_argument("image", help="girdi görselinin yolu")
    p_comb.add_argument("text", help='değişiklik, ör. "kırmızı" veya "uzun kollu"')
    p_comb.add_argument("-n", type=positive_int, default=5, help="sonuç sayısı (varsayılan: 5)")
    p_comb.add_argument("-w", type=weight, default=0.5,
                        help="metin ağırlığı: 0 = yalnız görsel, 1 = yalnız metin (varsayılan: 0.5)")
    p_comb.set_defaults(func=cmd_combine)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
