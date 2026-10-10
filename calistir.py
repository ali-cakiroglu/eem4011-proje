"""Aşamaları sırayla ölçer: parametreleri 01-20.png'de seçer, sonucu 21-100.png'de
bir kez raporlar.

Kullanım:
    python bitewing/calistir.py
    python bitewing/calistir.py --k 10
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import bitewing as bw

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

MESAFELER = [10, 20, 30, 45, 60]
PENCERELER = [31, 51, 81, 121, 161, 201]
MEDYANLAR = [3, 5, 7]
RASTGELE_TEKRAR = 50


def hazirla(bolum: list[str], kutular: dict):
    """Bölümün görüntülerini, diş maskelerini ve E0 kutularını bir kez yükler.

    E0'ın hiç lezyon işaretlemediği görüntüler atlanır: duyarlılık yalnız E0'ın
    kutuları üzerinden ölçüldüğü için bu görüntüler ne paya ne paydaya katkı
    verir, atlamak sonucu değiştirmez.
    """
    kullanilan = [d for d in bolum if kutular.get((d, bw.REFERANS))]
    atlanan = len(bolum) - len(kullanilan)
    print(f"    {len(kullanilan)} görüntü kullanılıyor"
          f" ({atlanan} görüntüde E0 hiç lezyon işaretlememiş, duyarlılığa katkısı yok)")
    g, maske, e0 = [], [], []
    for i, dosya in enumerate(kullanilan, 1):
        resim = bw.oku(dosya)
        g.append(resim)
        maske.append(bw.dis_maskesi(resim))
        e0.append(kutular[(dosya, bw.REFERANS)])
        if i % 20 == 0:
            print(f"    {i}/{len(kullanilan)} yüklendi", flush=True)
    return g, maske, e0


def olc(skorlar, maskeler, e0ler, mesafe: int, k: int, maske_uygula: bool) -> np.ndarray:
    """Duyarlılık eğrisi: k = 1..k için bulunan E0 kutularının oranı."""
    toplam = np.zeros(k, dtype=np.int64)
    hedef = 0
    for skor, maske, e0 in zip(skorlar, maskeler, e0ler):
        nokta = bw.adaylar(skor, k, mesafe, maske if maske_uygula else None)
        toplam += bw.profil(nokta, e0, k)
        hedef += len(e0)
    return toplam / hedef if hedef else toplam.astype(float)


def en_iyi_mesafe(skorlar, maskeler, e0ler, k: int, maske_uygula: bool, etiket: str):
    """Aday arası en az mesafeyi 01-20'de seçer."""
    satir = {}
    for m in MESAFELER:
        egri = olc(skorlar, maskeler, e0ler, m, k, maske_uygula)
        satir[m] = egri[-1]
        print(f"    {etiket} mesafe={m:>3}  Duyarlılık@{k} = {egri[-1]:.3f}", flush=True)
    iyi = max(satir, key=satir.get)
    print(f"    -> seçilen mesafe = {iyi} ({satir[iyi]:.3f})", flush=True)
    return iyi, satir[iyi]


def main():
    ap = argparse.ArgumentParser(description="Bitewing çürük adayı ölçümü")
    ap.add_argument("--k", type=int, default=10, help="görüntü başına aday sayısı")
    a = ap.parse_args()
    k = a.k
    t0 = time.time()

    kutular = bw.yukle_etiketler()
    secim = {}

    print("=" * 70)
    print(f"PARAMETRE BÖLÜMÜ ({len(bw.PARAMETRE)} görüntü: 01-20.png)")
    print("=" * 70)
    gP, maskeP, e0P = hazirla(bw.PARAMETRE, kutular)
    print(f"  E0 kutusu: {sum(len(x) for x in e0P)}")

    # --- A0 ham koyuluk ---------------------------------------------------- #
    print("\n[A0] ham koyuluk")
    skor = [bw.skor_a0(g) for g in gP]
    secim["A0"] = {"mesafe": en_iyi_mesafe(skor, maskeP, e0P, k, False, "A0")[0]}

    # --- A1 yerel arka plan ------------------------------------------------ #
    print("\n[A1] yerel arka plan çıkarma — pencere taraması")
    en_iyi = (-1.0, None, None)
    for p in PENCERELER:
        skor = [bw.skor_a1(g, p) for g in gP]
        m, deger = en_iyi_mesafe(skor, maskeP, e0P, k, False, f"pencere={p:>3}")
        if deger > en_iyi[0]:
            en_iyi = (deger, p, m)
    secim["A1"] = {"pencere": en_iyi[1], "mesafe": en_iyi[2]}
    print(f"  => A1 seçimi: pencere={en_iyi[1]}, mesafe={en_iyi[2]} ({en_iyi[0]:.3f})")

    # --- A2 medyan filtre -------------------------------------------------- #
    print("\n[A2] medyan filtre taraması")
    pencere = secim["A1"]["pencere"]
    en_iyi = (-1.0, None, None)
    for md in MEDYANLAR:
        skor = [bw.skor_a2(g, pencere, md) for g in gP]
        m, deger = en_iyi_mesafe(skor, maskeP, e0P, k, False, f"medyan={md}")
        if deger > en_iyi[0]:
            en_iyi = (deger, md, m)
    secim["A2"] = {"pencere": pencere, "medyan": en_iyi[1], "mesafe": en_iyi[2]}
    print(f"  => A2 seçimi: medyan={en_iyi[1]}, mesafe={en_iyi[2]} ({en_iyi[0]:.3f})")

    # --- A3 Otsu diş içi kısıtı -------------------------------------------- #
    print("\n[A3] Otsu diş içi kısıtı (A2 skoru + maske)")
    medyan = secim["A2"]["medyan"]
    skor = [bw.skor_a2(g, pencere, medyan) for g in gP]
    m, deger = en_iyi_mesafe(skor, maskeP, e0P, k, True, "A3")
    secim["A3"] = {"pencere": pencere, "medyan": medyan, "mesafe": m}
    print(f"  => A3 seçimi: mesafe={m} ({deger:.3f})")

    del gP, maskeP, e0P, skor

    # --- SONUÇ BÖLÜMÜ ------------------------------------------------------ #
    print("\n" + "=" * 70)
    print(f"SONUÇ BÖLÜMÜ ({len(bw.SONUC)} görüntü: 21-100.png) — parametreler dondurulmuş")
    print("=" * 70)
    gS, maskeS, e0S = hazirla(bw.SONUC, kutular)
    hedef = sum(len(x) for x in e0S)
    print(f"  E0 kutusu: {hedef}")

    egriler = {}

    print("\n  ölçülüyor: A0")
    egriler["A0"] = olc([bw.skor_a0(g) for g in gS], maskeS, e0S,
                        secim["A0"]["mesafe"], k, False)

    print("  ölçülüyor: A1")
    egriler["A1"] = olc([bw.skor_a1(g, secim["A1"]["pencere"]) for g in gS],
                        maskeS, e0S, secim["A1"]["mesafe"], k, False)

    print("  ölçülüyor: A2 ve A3 (aynı skor haritası)")
    skorS = [bw.skor_a2(g, pencere, medyan) for g in gS]
    egriler["A2"] = olc(skorS, maskeS, e0S, secim["A2"]["mesafe"], k, False)
    egriler["A3"] = olc(skorS, maskeS, e0S, secim["A3"]["mesafe"], k, True)
    del skorS

    print(f"  ölçülüyor: rastgele işaretleme ({RASTGELE_TEKRAR} tekrar)")
    uret = np.random.default_rng(0)
    birikim = np.zeros(k)
    for _ in range(RASTGELE_TEKRAR):
        toplam = np.zeros(k, dtype=np.int64)
        for maske, e0 in zip(maskeS, e0S):
            toplam += bw.profil(bw.rastgele_adaylar(maske, k, uret), e0, k)
        birikim += toplam / hedef
    egriler["rastgele"] = birikim / RASTGELE_TEKRAR

    print("  ölçülüyor: 7 hekim")
    hekim = {h: bw.hekim_duyarliligi(kutular, bw.SONUC, h) for h in bw.HEKIMLER}

    # --- Rapor ------------------------------------------------------------- #
    print("\n" + "=" * 70)
    print(f"SONUÇLAR — 21-100.png, {hedef} E0 kutusu, görüntü başına {k} aday")
    print("=" * 70)
    print(f"\n{'Aşama':<34}{'Duyarlılık@' + str(k):>14}")
    print("-" * 48)
    adlar = {
        "rastgele": "rastgele işaretleme (alt sınır)",
        "A0": "A0  ham koyuluk",
        "A1": "A1  + yerel arka plan çıkarma",
        "A2": "A2  + medyan filtre",
        "A3": "A3  + Otsu diş içi kısıtı",
    }
    for ad in ["rastgele", "A0", "A1", "A2", "A3"]:
        print(f"{adlar[ad]:<34}{egriler[ad][-1]:>14.3f}")

    print(f"\n{'Hekim':<10}{'işaret/görüntü':>16}{'Duyarlılık':>12}")
    print("-" * 38)
    for h, (duy, isaret) in hekim.items():
        print(f"{h:<10}{isaret:>16.2f}{duy:>12.3f}")

    print(f"\nDuyarlılık eğrisi (aday sayısı 1..{k}):")
    print(f"{'aday':<6}" + "".join(f"{ad:>10}" for ad in ["rastgele", "A0", "A1", "A2", "A3"]))
    for i in range(k):
        print(f"{i + 1:<6}" + "".join(f"{egriler[ad][i]:>10.3f}"
                                     for ad in ["rastgele", "A0", "A1", "A2", "A3"]))

    cikti = {
        "k": k,
        "parametre_bolumu": bw.PARAMETRE,
        "sonuc_bolumu": bw.SONUC,
        "e0_kutusu_sonuc": hedef,
        "secilen_parametreler": secim,
        "egriler": {ad: [float(v) for v in egriler[ad]] for ad in egriler},
        "duyarlilik_at_k": {ad: float(egriler[ad][-1]) for ad in egriler},
        "hekimler": {h: {"duyarlilik": d, "isaret_per_goruntu": i}
                     for h, (d, i) in hekim.items()},
    }
    yol = Path(__file__).resolve().parent / "sonuclar.json"
    yol.write_text(json.dumps(cikti, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nKaydedildi: {yol}")
    print(f"Süre: {time.time() - t0:.0f} saniye")


if __name__ == "__main__":
    main()
