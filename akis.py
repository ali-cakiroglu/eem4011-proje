"""Şekil 1: yöntemin akış şeması, gerçek ara görüntülerle.

Bir röntgen (varsayılan 02.png) dört aşamadan geçirilir; her panelde o aşamanın
skor haritası, kendi seçtiği 10 aday (mavi +) ve E0 kutuları (sarı: yakalandı,
kırmızı: kaçırıldı) gösterilir. Son panel ölçüm kuralını ve test sonuçlarını özetler.

Kullanım:
    python bitewing/akis.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bitewing as bw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BURASI = Path(__file__).resolve().parent
BULUNAN, KACAN, ADAY = "#f2c200", "#e02020", "#2f7fe0"


def v(x, b=3):
    """Sayıyı Türkçe ondalık virgülle yazar."""
    return f"{x:.{b}f}".replace(".", ",")


def goster(eksen, harita, cmap):
    alt, ust = np.percentile(harita, [1, 99])
    eksen.imshow(harita, cmap=cmap, vmin=alt, vmax=ust)
    eksen.set_xticks([])
    eksen.set_yticks([])


def isaretle(eksen, noktalar, e0):
    bulundu = [False] * len(e0)
    for x, y in noktalar:
        for i, (bx, by, bw_, bh) in enumerate(e0):
            if not bulundu[i] and bx <= x <= bx + bw_ and by <= y <= by + bh:
                bulundu[i] = True
                break
    for (x, y, w, h), var in zip(e0, bulundu):
        eksen.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=1.6,
                                  edgecolor=BULUNAN if var else KACAN))
    for x, y in noktalar:
        eksen.plot(x, y, marker="+", markersize=9, markeredgewidth=2, color=ADAY)
    return sum(bulundu)


def main():
    dosya = "02.png"
    s = json.loads((BURASI / "sonuclar.json").read_text(encoding="utf-8"))
    secim, duy = s["secilen_parametreler"], s["duyarlilik_at_k"]
    hekim = [d["duyarlilik"] for d in s["hekimler"].values()]

    kutular = bw.yukle_etiketler()
    e0 = kutular[(dosya, bw.REFERANS)]
    g = bw.oku(dosya)
    maske = bw.dis_maskesi(g)
    p = secim["A2"]["pencere"]
    m = secim["A2"]["medyan"]

    a0 = bw.skor_a0(g)
    a1 = bw.skor_a1(g, p)
    a2 = bw.skor_a2(g, p, m)
    a3 = np.where(maske, a2, a2[maske].min())

    paneller = [
        ("1 · Röntgen (02.png)", g, "gray", None, None),
        ("2 · A0 — ham koyuluk", a0, "magma",
         bw.adaylar(a0, 10, secim["A0"]["mesafe"]), "A0"),
        (f"3 · A1 — yerel arka plan çıkarma\n(ortalama filtre {p}×{p})", a1, "magma",
         bw.adaylar(a1, 10, secim["A1"]["mesafe"]), "A1"),
        (f"4 · A2 — medyan filtre ({m}×{m})", a2, "magma",
         bw.adaylar(a2, 10, secim["A2"]["mesafe"]), "A2"),
        ("5 · A3 — Otsu ile diş içi kısıtı", a3, "magma",
         bw.adaylar(a2, 10, secim["A3"]["mesafe"], maske), "A3"),
    ]

    fig, eksenler = plt.subplots(2, 3, figsize=(16, 9.6))
    eksenler = eksenler.ravel()

    for eksen, (baslik, harita, cmap, noktalar, ad) in zip(eksenler, paneller):
        goster(eksen, harita, cmap)
        if noktalar is None:
            for x, y, w, h in e0:
                eksen.add_patch(Rectangle((x, y), w, h, fill=False,
                                          linewidth=1.6, edgecolor=BULUNAN))
            alt = f"E0'ın {len(e0)} kutusu"
        else:
            k = isaretle(eksen, noktalar, e0)
            alt = (f"02.png: {k}/{len(e0)} kutu yakalandı   ·   "
                   f"test Duyarlılık@10 = {v(duy[ad])}")
        eksen.set_title(baslik, fontsize=11.5, fontweight="bold")
        eksen.set_xlabel(alt, fontsize=9.5)

    son = eksenler[5]
    son.axis("off")
    metin = (
        "Ölçüm\n\n"
        "Her aşama 10 aday seçer: en yüksek tepe → çevresini kapat → tekrarla.\n\n"
        "Bir adayın merkezi E0 kutusunun içine (kenar dahil)\n"
        "düşerse kutu yakalanmış sayılır; her kutu bir kez.\n\n"
        "Duyarlılık@10 = yakalanan E0 kutusu / toplam E0 kutusu\n\n"
        "Test bölümü (21–100.png, 218 E0 kutusu):\n"
        f"   rastgele işaretleme   {v(duy['rastgele'])}\n"
        f"   A0  {v(duy['A0'])}   →   A1  {v(duy['A1'])}   →   "
        f"A2  {v(duy['A2'])}   →   A3  {v(duy['A3'])}\n"
        f"   7 hekim (aynı kural)   {v(min(hekim), 2)} – {v(max(hekim), 2)}"
    )
    son.text(0.02, 0.97, metin, va="top", ha="left", fontsize=10.5,
             family="DejaVu Sans", transform=son.transAxes,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#f4f6fa",
                       edgecolor="#9aa5b8"))
    son.plot([], [], marker="s", linestyle="", markersize=9, markerfacecolor="none",
             markeredgecolor=BULUNAN, markeredgewidth=2, label="E0 kutusu — yakalandı")
    son.plot([], [], marker="s", linestyle="", markersize=9, markerfacecolor="none",
             markeredgecolor=KACAN, markeredgewidth=2, label="E0 kutusu — kaçırıldı")
    son.plot([], [], marker="+", linestyle="", markersize=11, markeredgewidth=2,
             color=ADAY, label="o aşamanın 10 adayı")
    son.legend(loc="lower left", fontsize=9.5, frameon=True)

    fig.suptitle("Şekil 1 — Yöntemin akışı: her aşamanın skor haritası ve seçtiği 10 aday",
                 fontsize=14, fontweight="bold")
    fig.text(0.5, 0.008,
             "Örnek görüntü 02.png karar bölümündendir (01–20); sayılar test bölümüne aittir. "
             "Veri: Kybic, Tichý, Kunt, Mendeley Data, DOI: 10.17632/4fbdxs7s7w.1, CC BY-NC 3.0.",
             ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.025, 1, 0.95))

    # paneller arasına oklar
    sira = [0, 1, 2, 3, 4]
    for a, b in zip(sira, sira[1:]):
        pa, pb = eksenler[a].get_position(), eksenler[b].get_position()
        if abs(pa.y0 - pb.y0) > 0.01:      # satır atlayan geçişte ok yok; numaralar sırayı veriyor
            continue
        bas, son_ = (pa.x1 + 0.003, (pa.y0 + pa.y1) / 2), (pb.x0 - 0.003, (pb.y0 + pb.y1) / 2)
        fig.add_artist(FancyArrowPatch(bas, son_, transform=fig.transFigure,
                                       arrowstyle="-|>", mutation_scale=18,
                                       linewidth=2, color="#444444",
                                       connectionstyle="arc3,rad=0.0"))

    yol = BURASI / "sekil_akis.png"
    fig.savefig(yol, dpi=150)
    print(f"Kaydedildi: {yol}")


if __name__ == "__main__":
    main()
