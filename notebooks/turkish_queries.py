"""Türkçe metinden arama için ayrık değerlendirme sorguları.

Bu set Türkçe moda sözlüğü (fashion_rec/glossary.py) yazılmadan ÖNCE
hazırlandı ve sözlük ayarlanırken kullanılmadı; sözlüğün gerçek etkisini
ölçmek içindir. `evaluate_products.TEXT_QUERIES` ise geliştirme setidir
(sözlük yazılırken görülmüştü).

Biçim: (sorgu, {etiket alanı: kabul edilen değerler}); değerler
metadata.csv ile birebir aynı olmalıdır.
"""

HELD_OUT_QUERIES = [
    ("lacivert gömlek", {"articleType": {"Shirts"}, "baseColour": {"Navy Blue"}}),
    ("beyaz tişört", {"articleType": {"Tshirts"}, "baseColour": {"White"}}),
    ("sarı tişört", {"articleType": {"Tshirts"}, "baseColour": {"Yellow"}}),
    ("yeşil tişört", {"articleType": {"Tshirts"}, "baseColour": {"Green"}}),
    ("mor elbise", {"articleType": {"Dresses"}, "baseColour": {"Purple"}}),
    ("pembe çanta", {"articleType": {"Handbags", "Clutches"}, "baseColour": {"Pink"}}),
    ("gri eşofman altı", {"articleType": {"Track Pants"}, "baseColour": {"Grey", "Grey Melange"}}),
    ("kahverengi kemer", {"articleType": {"Belts"}, "baseColour": {"Brown", "Coffee Brown", "Tan"}}),
    ("siyah cüzdan", {"articleType": {"Wallets"}, "baseColour": {"Black"}}),
    ("gümüş yüzük", {"articleType": {"Ring"}, "baseColour": {"Silver"}}),
    ("altın kolye", {"articleType": {"Necklace and Chains", "Pendant"}, "baseColour": {"Gold"}}),
    ("kırmızı ruj", {"articleType": {"Lipstick"}, "baseColour": {"Red", "Maroon"}}),
    ("erkek kemer", {"articleType": {"Belts"}, "gender": {"Men"}}),
    ("kadın sandalet", {"articleType": {"Sandals", "Sports Sandals", "Flats"}, "gender": {"Women"}}),
    ("kadın tayt", {"articleType": {"Leggings", "Jeggings", "Tights"}, "gender": {"Women"}}),
    ("erkek şort", {"articleType": {"Shorts", "Lounge Shorts"}, "gender": {"Men"}}),
    ("kız çocuk tişört", {"articleType": {"Tshirts", "Tops"}, "gender": {"Girls"}}),
    ("erkek ceket", {"articleType": {"Jackets", "Blazers", "Nehru Jackets", "Rain Jacket"}, "gender": {"Men"}}),
    ("kadın saat", {"articleType": {"Watches"}, "gender": {"Women"}}),
    ("oje", {"articleType": {"Nail Polish"}}),
    ("şapka", {"articleType": {"Caps", "Hat"}}),
    ("atkı", {"articleType": {"Scarves", "Stoles", "Mufflers"}}),
    ("sütyen", {"articleType": {"Bra"}}),
    ("terlik", {"articleType": {"Flip Flops"}}),
    ("kazak", {"articleType": {"Sweaters", "Sweatshirts"}}),
    ("kravat", {"articleType": {"Ties", "Ties and Cufflinks"}}),
    ("bilezik", {"articleType": {"Bangle", "Bracelet"}}),
    ("eşofman üstü", {"articleType": {"Sweatshirts", "Jackets", "Tracksuits"}}),
    ("gece elbisesi", {"articleType": {"Nightdress", "Night suits", "Dresses"}}),
    ("dizüstü bilgisayar çantası", {"articleType": {"Laptop Bag"}}),
]
