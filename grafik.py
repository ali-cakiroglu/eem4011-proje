"""Duyarlılık–aday sayısı grafiğini çizer.

Sol panel: çalışma noktası (1-10 aday) — aşamalar, rastgele alt sınır ve 7 hekim.
Sağ panel: yöntemin tavanı (1-200 aday) — A3 hekim bandına nerede yetişiyor.

Kullanım:
    python bitewing/grafik.py
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bitewing as bw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

BURASI = Path(__file__).resolve().parent
K_UZUN = 200
RENK = {"rastgele": "#8c8c8c", "A0": "#c6a700", "A1": "#e07b39",
        "A2": "#3b7dd8", "A3": "#1a7f37"}
ETIKET = {"rastgele": "Rastgele işaretleme", "A0": "A0  ham koyuluk",
          "A1": "A1  + yerel arka plan", "A2": "A2  + medyan filtre",
          "A3": "A3  + Otsu diş içi kısıtı"}


def uzun_egri(secim: dict) -> tuple[np.ndarray, np.ndarray]:
    """A3 ve rastgele için 200 adaya kadar eğri (önbelleğe alınır)."""
    onbellek = BURASI / "uzun_egri.json"
    if onbellek.exists():
        d = json.loads(onbellek.read_text(encoding="utf-8"))
        return np.array(d["A3"]), np.array(d["rastgele"])

    kutular = bw.yukle_etiketler()
    bolum = [d for d in bw.SONUC if kutular.get((d, bw.REFERANS))]
    toplam = np.zeros(K_UZUN, dtype=np.int64)
    rast = np.zeros(K_UZUN)
    hedef = 0
    uret = np.random.default_rng(0)
    for i, dosya in enumerate(bolum, 1):
        g = bw.oku(dosya)
        maske = bw.dis_maskesi(g)
        e0 = kutular[(dosya, bw.REFERANS)]
        hedef += len(e0)
        skor = bw.skor_a2(g, secim["pencere"], secim["medyan"])
        toplam += bw.profil(bw.adaylar(skor, K_UZUN, secim["mesafe"], maske), e0, K_UZUN)
        rast += bw.profil(bw.rastgele_adaylar(maske, K_UZUN, uret), e0, K_UZUN)
        if i % 20 == 0:
            print(f"  uzun eğri {i}/{len(bolum)}", flush=True)
    a3, rs = toplam / hedef, rast / hedef
    onbellek.write_text(json.dumps({"A3": a3.tolist(), "rastgele": rs.tolist()},
                                   indent=2), encoding="utf-8")
    return a3, rs


def main():
    s = json.loads((BURASI / "sonuclar.json").read_text(encoding="utf-8"))
    k = s["k"]
    egri = {ad: np.array(v) for ad, v in s["egriler"].items()}
    hekim = s["hekimler"]
    x = np.arange(1, k + 1)

    fig, (sol, sag) = plt.subplots(1, 2, figsize=(13.5, 5.4))

    # --- Sol panel: çalışma noktası ---------------------------------------- #
    for ad in ["A3", "A2", "A1", "A0", "rastgele"]:
        sol.plot(x, egri[ad], marker="o", markersize=3.5, linewidth=2,
                 color=RENK[ad], label=ETIKET[ad],
                 linestyle="--" if ad == "rastgele" else "-")
    for ad, d in hekim.items():
        sol.scatter(d["isaret_per_goruntu"], d["duyarlilik"], s=60,
                    marker="D", color="#b3306e", zorder=5)
        sol.annotate(ad, (d["isaret_per_goruntu"], d["duyarlilik"]),
                     textcoords="offset points", xytext=(7, -3), fontsize=9,
                     color="#b3306e")
    sol.scatter([], [], s=60, marker="D", color="#b3306e", label="Hekimler (E1-E4, N1-N3)")
    sol.set_xlabel("Görüntü başına işaret sayısı")
    sol.set_ylabel("Duyarlılık (E0 kutularının bulunan oranı)")
    sol.set_title("Çalışma noktası: görüntü başına 10 aday", fontsize=11)
    sol.set_xlim(0, k + 0.6)
    sol.set_ylim(0, 1.0)
    sol.grid(alpha=0.25)
    sol.legend(fontsize=8.5, loc="center left")

    # --- Sağ panel: tavan --------------------------------------------------- #
    a3u, rsu = uzun_egri(s["secilen_parametreler"]["A3"])
    xu = np.arange(1, len(a3u) + 1)
    alt = min(d["duyarlilik"] for d in hekim.values())
    ust = max(d["duyarlilik"] for d in hekim.values())
    sag.axhspan(alt, ust, color="#b3306e", alpha=0.13)
    sag.text(len(a3u) * 0.42, (alt + ust) / 2, "hekimlerin aralığı",
             color="#b3306e", fontsize=9, va="center")
    sag.plot(xu, a3u, linewidth=2, color=RENK["A3"], label="A3  + Otsu diş içi kısıtı")
    sag.plot(xu, rsu, linewidth=2, linestyle="--", color=RENK["rastgele"],
             label="Rastgele işaretleme")
    sag.axvline(k, color="#444444", linewidth=1, linestyle=":")
    sag.text(k + 4, 0.03, f"çalışma noktası\n({k} aday)", fontsize=8.5, color="#444444")
    ort_hekim = np.mean([d["isaret_per_goruntu"] for d in hekim.values()])
    sag.axvline(ort_hekim, color="#b3306e", linewidth=1, linestyle=":")
    sag.set_xlabel("Görüntü başına işaret sayısı")
    sag.set_ylabel("Duyarlılık")
    sag.set_title("Yöntemin tavanı: hekim düzeyi ~150-200 adayda yakalanıyor", fontsize=11)
    sag.set_xlim(0, len(a3u))
    sag.set_ylim(0, 1.0)
    sag.grid(alpha=0.25)
    sag.legend(fontsize=8.5, loc="lower right")

    fig.suptitle("Bitewing röntgende klasik yöntemle çürük adayı işaretleme — "
                 "21-100.png, 218 E0 kutusu", fontsize=12.5)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    yol = BURASI / "grafik.png"
    fig.savefig(yol, dpi=160)
    print(f"Kaydedildi: {yol}")
    for kk in (10, 50, 100, 150, 200):
        print(f"  A3@{kk:<4} = {a3u[kk - 1]:.3f}    rastgele@{kk:<4} = {rsu[kk - 1]:.3f}")


if __name__ == "__main__":
    main()
