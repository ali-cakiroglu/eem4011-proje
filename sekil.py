"""Tek bir röntgenin üstüne E0 kutularını ve yöntemin adaylarını bindirir.

Sarı kutu  : E0'ın işaretlediği ve yöntemin bulduğu lezyon
Kırmızı kutu: E0'ın işaretlediği ama yöntemin kaçırdığı lezyon
Mavi +     : yöntemin seçtiği 10 aday
Başlık     : kaç E0 kutusunun bulunduğu / toplam kaç E0 kutusu olduğu

Kullanım:
    python bitewing/sekil.py            # 02.png
    python bitewing/sekil.py --goruntu 27
"""

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bitewing as bw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BURASI = Path(__file__).resolve().parent
BULUNAN, KACAN, ADAY = "#f2c200", "#e02020", "#2f7fe0"


def eslesme(noktalar, kutular) -> list[bool]:
    """Her E0 kutusu için: 10 adaydan biri içine düştü mü? (merkez-kutu kuralı)"""
    kullanildi = [False] * len(kutular)
    for x, y in noktalar:
        for i, (bx, by, bw_, bh) in enumerate(kutular):
            if not kullanildi[i] and bx <= x <= bx + bw_ and by <= y <= by + bh:
                kullanildi[i] = True
                break
    return kullanildi


def main():
    ap = argparse.ArgumentParser(description="Örnek röntgen şekli")
    ap.add_argument("--goruntu", default="02", help="görüntü numarası, ör. 02")
    a = ap.parse_args()
    dosya = f"{a.goruntu}.png"

    secim = json.loads((BURASI / "sonuclar.json").read_text(encoding="utf-8")
                       )["secilen_parametreler"]["A3"]
    kutular = bw.yukle_etiketler()
    e0 = kutular.get((dosya, bw.REFERANS), [])
    if not e0:
        sys.exit(f"{dosya} için E0 kutusu yok, başka görüntü seçin.")

    g = bw.oku(dosya)
    maske = bw.dis_maskesi(g)
    skor = bw.skor_a2(g, secim["pencere"], secim["medyan"])
    noktalar = bw.adaylar(skor, 10, secim["mesafe"], maske)
    bulundu = eslesme(noktalar, e0)

    fig, eksen = plt.subplots(figsize=(11, 8.7))
    eksen.imshow(g, cmap="gray", vmin=0, vmax=1)

    for (x, y, w, h), var in zip(e0, bulundu):
        eksen.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2.2,
                                  edgecolor=BULUNAN if var else KACAN))
    for i, (x, y) in enumerate(noktalar, 1):
        eksen.plot(x, y, marker="+", markersize=13, markeredgewidth=2.4, color=ADAY)
        eksen.annotate(str(i), (x, y), textcoords="offset points", xytext=(8, 5),
                       fontsize=9, color=ADAY, fontweight="bold")

    eksen.plot([], [], marker="s", linestyle="", markersize=9,
               markerfacecolor="none", markeredgecolor=BULUNAN, markeredgewidth=2,
               label="E0 kutusu — bulundu")
    eksen.plot([], [], marker="s", linestyle="", markersize=9,
               markerfacecolor="none", markeredgecolor=KACAN, markeredgewidth=2,
               label="E0 kutusu — kaçırıldı")
    eksen.plot([], [], marker="+", linestyle="", markersize=12, markeredgewidth=2.4,
               color=ADAY, label="yöntemin adayı (1-10)")
    eksen.legend(loc="lower right", fontsize=9, framealpha=0.9)

    eksen.set_title(f"{dosya} — A3 yöntemi, 10 aday · "
                    f"E0 kutusu: {sum(bulundu)} bulundu / {len(e0)} toplam",
                    fontsize=12)
    eksen.set_xticks([])
    eksen.set_yticks([])
    fig.tight_layout()

    yol = BURASI / f"sekil1_{a.goruntu}.png"
    fig.savefig(yol, dpi=150)
    print(f"Kaydedildi: {yol}")
    print(f"  E0 kutusu : {len(e0)}")
    print(f"  bulunan   : {sum(bulundu)}")
    print(f"  aday      : {len(noktalar)}")


if __name__ == "__main__":
    main()
