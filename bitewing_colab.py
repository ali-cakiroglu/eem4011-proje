"""Bitewing roentgende klasik yontemle curuk adayi isaretleme.

bitewing_colab.ipynb defterinin duz Python hali.
"""


# ----------------------------------------------------------------------
# # Bitewing röntgende klasik yöntemle çürük adayı işaretleme
# 
# Her röntgende **10 çürük adayı** işaretleyip uzman **E0**'ın kutularına göre
# **Duyarlılık@10** ölçülür. Hiçbir aşama veriden öğrenmez: yalnız ortalama filtre,
# medyan filtre ve Otsu eşiği kullanılır.
# 
# | Aşama | Ne ekliyor |
# |---|---|
# | A0 | ham koyuluk |
# | A1 | + yerel arka plan çıkarma (ortalama filtre) |
# | A2 | + medyan filtre |
# | A3 | + Otsu eşiğiyle diş içi kısıtı |
# 
# Parametreler yalnız `01-20.png` üzerinde seçilir, sonuç `21-100.png` üzerinde
# bir kez ölçülür.
# 
# **Veri:** Dental caries in bitewing radiographs — Mendeley Data, CC BY-NC 3.0,
# https://data.mendeley.com/datasets/4fbdxs7s7w/1
# ----------------------------------------------------------------------


# ----------------------------------------------------------------------
# ## 1. Veriyi indir
# 
# Veri depoya konulmaz, her çalıştırmada Mendeley'den iner (~75 MB).
# ----------------------------------------------------------------------

import hashlib, pathlib, urllib.request, zipfile

URL = ("https://data.mendeley.com/public-files/datasets/4fbdxs7s7w/"
       "files/66498986-c8ec-4285-9eb7-4f56f65463c6/file_downloaded")
SHA256 = "3bbe54645aab48252edd53f32729df6e8e6509c20781fd30cabdfb3e6ec9ca78"
ZIP = pathlib.Path("dental_rtg_test.zip")

if not ZIP.exists():
    print("indiriliyor...")
    urllib.request.urlretrieve(URL, ZIP)

ozet = hashlib.sha256(ZIP.read_bytes()).hexdigest()
print("SHA-256 dogru" if ozet == SHA256 else f"UYARI: ozet uyusmuyor ({ozet})")

if not pathlib.Path("dental_rtg_test/images").exists():
    with zipfile.ZipFile(ZIP) as z:
        z.extractall(".")
print(sum(1 for _ in pathlib.Path("dental_rtg_test/images").iterdir()), "goruntu")


# ----------------------------------------------------------------------
# ## 2. Temel işlemler
# 
# Üçü de `numpy` ile yazıldı; ek kütüphane gerekmiyor.
# ----------------------------------------------------------------------

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

KOK = Path("dental_rtg_test")
GORUNTU, ETIKET = KOK / "images", KOK / "test_annotations_anonymized.json"
REFERANS, HEKIMLER = "E0", ["E1", "E2", "E3", "E4", "N1", "N2", "N3"]
PARAMETRE = [f"{i:02d}.png" for i in range(1, 21)]
SONUC = [f"{i:02d}.png" for i in range(21, 101)]


def ortalama_filtre(g, k):
    """k x k ortalama filtre, integral goruntuyle."""
    r = k // 2
    p = np.pad(g, r, mode="reflect").astype(np.float64)
    bt = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = g.shape
    t = bt[k:k+H, k:k+W] - bt[0:H, k:k+W] - bt[k:k+H, 0:W] + bt[0:H, 0:W]
    return (t / (k * k)).astype(np.float32)


def medyan_filtre(g, k):
    """k x k medyan filtre."""
    r = k // 2
    p = np.pad(g, r, mode="reflect")
    return np.median(sliding_window_view(p, (k, k)), axis=(-2, -1)).astype(np.float32)


def otsu_esigi(g, kutu=256):
    """Otsu (1979): siniflar arasi varyansi en cok yapan esik."""
    h, kenar = np.histogram(g, bins=kutu, range=(0.0, 1.0))
    h = h.astype(np.float64)
    merkez = (kenar[:-1] + kenar[1:]) / 2.0
    a0 = h.cumsum()
    a1 = h.sum() - a0
    kum = (h * merkez).cumsum()
    m0 = np.divide(kum, a0, out=np.zeros_like(kum), where=a0 > 0)
    m1 = np.divide(kum[-1] - kum, a1, out=np.zeros_like(kum), where=a1 > 0)
    arasi = a0 * a1 * (m0 - m1) ** 2
    arasi[(a0 == 0) | (a1 == 0)] = 0.0
    return float(merkez[int(np.argmax(arasi))])


def dis_maskesi(g):
    """Otsu esiginin ustundeki parlak bolge: dis."""
    return g > otsu_esigi(g)


# ----------------------------------------------------------------------
# ## 3. Veriyi oku
# 
# Her işaretleyici COCO biçiminde ayrı bir `category`.
# ----------------------------------------------------------------------

def yukle_etiketler():
    """(dosya, isaretleyici) -> [(x, y, genislik, yukseklik), ...]"""
    d = json.loads(ETIKET.read_text(encoding="utf-8"))
    isim = {c["id"]: c["name"] for c in d["categories"]}
    dosya = {im["id"]: im["file_name"] for im in d["images"]}
    kutular = {}
    for a in d["annotations"]:
        kutular.setdefault((dosya[a["image_id"]], isim[a["category_id"]]), []) \
               .append(tuple(float(v) for v in a["bbox"]))
    return kutular


def oku(dosya):
    return np.asarray(Image.open(GORUNTU / dosya).convert("L"), dtype=np.float32) / 255.0


kutular = yukle_etiketler()
say = {h: sum(len(v) for (d, i), v in kutular.items() if i == h)
       for h in [REFERANS] + HEKIMLER}
print("isaretleyici basina kutu:", say)
print("E0 toplam:", say[REFERANS], "| hepsi:", sum(say.values()))


# ----------------------------------------------------------------------
# ## 4. Aşamaların skor haritaları ve aday seçimi
# 
# Skor haritası "burada lezyon olma ihtimali" gibi okunur. Adaylar, en yüksek tepeyi
# alıp çevresini kapatarak sırayla seçilir.
# ----------------------------------------------------------------------

def skor_a0(g):
    """Ham koyuluk."""
    return 1.0 - g


def skor_a1(g, pencere):
    """Yerel arka plan cikarma: cevresine gore ne kadar koyu."""
    return ortalama_filtre(g, pencere) - g


def skor_a2(g, pencere, medyan):
    """A1 + medyan filtre."""
    return medyan_filtre(skor_a1(g, pencere), medyan)


def adaylar(skor, n, en_az_mesafe, maske=None):
    """En yuksek tepeyi al, cevresini kapat, n kez tekrarla."""
    s = skor.astype(np.float32, copy=True)
    if maske is not None:
        s[~maske] = -np.inf
    H, W = s.shape
    r = int(en_az_mesafe)
    secilen = []
    for _ in range(n):
        y, x = divmod(int(np.argmax(s)), W)
        if not np.isfinite(s[y, x]):
            break
        secilen.append((x, y))
        s[max(0, y-r):y+r+1, max(0, x-r):x+r+1] = -np.inf
    return secilen


def rastgele_adaylar(maske, n, uret):
    """Alt sinir: dis maskesi icinden rastgele n nokta."""
    ys, xs = np.nonzero(maske)
    if len(ys) == 0:
        return []
    s = uret.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in s]


# ----------------------------------------------------------------------
# ## 5. Ölçüm
# 
# **Merkez-kutu kuralı:** bir işaretin merkezi bir E0 kutusunun içindeyse o kutu
# bulunmuş sayılır; her kutu en fazla bir kez sayılır. Aynı kural yönteme, rastgele
# işaretlemeye ve hekimlere birebir uygulanır.
# ----------------------------------------------------------------------

def profil(noktalar, kutular_, k_azami):
    """Ilk k isaret kullanildiginda bulunan E0 kutusu sayisi (k = 1..k_azami)."""
    kullanildi = [False] * len(kutular_)
    cikti = np.zeros(k_azami, dtype=np.int32)
    bulunan = 0
    for k in range(k_azami):
        if k < len(noktalar):
            x, y = noktalar[k]
            for i, (bx, by, bw_, bh) in enumerate(kutular_):
                if not kullanildi[i] and bx <= x <= bx+bw_ and by <= y <= by+bh:
                    kullanildi[i] = True
                    bulunan += 1
                    break
        cikti[k] = bulunan
    return cikti


def hazirla(bolum):
    """E0'in kutu cizdigi goruntuleri yukler (otekiler duyarliliga katki vermez)."""
    kullanilan = [d for d in bolum if kutular.get((d, REFERANS))]
    g = [oku(d) for d in kullanilan]
    return g, [dis_maskesi(x) for x in g], [kutular[(d, REFERANS)] for d in kullanilan]


def olc(skorlar, maskeler, e0ler, mesafe, k, maske_uygula):
    toplam, hedef = np.zeros(k, dtype=np.int64), 0
    for skor, maske, e0 in zip(skorlar, maskeler, e0ler):
        toplam += profil(adaylar(skor, k, mesafe, maske if maske_uygula else None), e0, k)
        hedef += len(e0)
    return toplam / hedef


# ----------------------------------------------------------------------
# ## 6. Parametre seçimi — yalnız 01-20.png
# ----------------------------------------------------------------------

K = 10
MESAFELER, PENCERELER, MEDYANLAR = [10, 20, 30, 45, 60], [31, 51, 81, 121, 161, 201], [3, 5, 7]

gP, maskeP, e0P = hazirla(PARAMETRE)
print(f"{len(gP)} goruntu, {sum(len(x) for x in e0P)} E0 kutusu")


def en_iyi_mesafe(skorlar, maske_uygula):
    d = {m: olc(skorlar, maskeP, e0P, m, K, maske_uygula)[-1] for m in MESAFELER}
    iyi = max(d, key=d.get)
    return iyi, d[iyi]

secim = {}
secim["A0"] = {"mesafe": en_iyi_mesafe([skor_a0(g) for g in gP], False)[0]}

en_iyi = (-1, None, None)
for p in PENCERELER:
    m, v = en_iyi_mesafe([skor_a1(g, p) for g in gP], False)
    print(f"A1 pencere={p:>3}: {v:.3f}")
    if v > en_iyi[0]:
        en_iyi = (v, p, m)
secim["A1"] = {"pencere": en_iyi[1], "mesafe": en_iyi[2]}

pencere = secim["A1"]["pencere"]
en_iyi = (-1, None, None)
for md_ in MEDYANLAR:
    m, v = en_iyi_mesafe([skor_a2(g, pencere, md_) for g in gP], False)
    print(f"A2 medyan={md_}: {v:.3f}")
    if v > en_iyi[0]:
        en_iyi = (v, md_, m)
secim["A2"] = {"pencere": pencere, "medyan": en_iyi[1], "mesafe": en_iyi[2]}

medyan = secim["A2"]["medyan"]
m, v = en_iyi_mesafe([skor_a2(g, pencere, medyan) for g in gP], True)
secim["A3"] = {"pencere": pencere, "medyan": medyan, "mesafe": m}
print("\nsecilen:", secim)


# ----------------------------------------------------------------------
# ## 7. Sonuç — 21-100.png, parametreler dondurulmuş
# ----------------------------------------------------------------------

gS, maskeS, e0S = hazirla(SONUC)
hedef = sum(len(x) for x in e0S)
print(f"{len(gS)} goruntu, {hedef} E0 kutusu")

egri = {}
egri["A0"] = olc([skor_a0(g) for g in gS], maskeS, e0S, secim["A0"]["mesafe"], K, False)
egri["A1"] = olc([skor_a1(g, secim["A1"]["pencere"]) for g in gS], maskeS, e0S,
                 secim["A1"]["mesafe"], K, False)
skorS = [skor_a2(g, pencere, medyan) for g in gS]
egri["A2"] = olc(skorS, maskeS, e0S, secim["A2"]["mesafe"], K, False)
egri["A3"] = olc(skorS, maskeS, e0S, secim["A3"]["mesafe"], K, True)

uret = np.random.default_rng(0)
birikim = np.zeros(K)
for _ in range(50):
    t = np.zeros(K, dtype=np.int64)
    for maske, e0 in zip(maskeS, e0S):
        t += profil(rastgele_adaylar(maske, K, uret), e0, K)
    birikim += t / hedef
egri["rastgele"] = birikim / 50

print(f"\n{'Asama':<34}{'Duyarlilik@10':>14}")
for ad in ["rastgele", "A0", "A1", "A2", "A3"]:
    print(f"{ad:<34}{egri[ad][-1]:>14.3f}")


# ----------------------------------------------------------------------
# ## 8. Hekimler — aynı kural, aynı bölüm
# ----------------------------------------------------------------------

def hekim_duyarliligi(bolum, hekim):
    bulunan = hedef_ = isaret = 0
    for d in bolum:
        e0 = kutular.get((d, REFERANS), [])
        merkez = [(x + w/2, y + h/2) for x, y, w, h in kutular.get((d, hekim), [])]
        hedef_ += len(e0)
        isaret += len(merkez)
        if e0 and merkez:
            bulunan += int(profil(merkez, e0, len(merkez))[-1])
    return bulunan / hedef_, isaret / len(bolum)


hekim = {h: hekim_duyarliligi(SONUC, h) for h in HEKIMLER}
print(f"{'Hekim':<8}{'isaret/goruntu':>16}{'Duyarlilik':>12}")
for h, (duy, isa) in hekim.items():
    print(f"{h:<8}{isa:>16.2f}{duy:>12.3f}")


# ----------------------------------------------------------------------
# ## 9. Şekiller
# 
# `sekil1_02.png` — örnek röntgen: sarı kutu E0'ın bulunan lezyonu, kırmızı kutu
# kaçırılan lezyon, mavi + yöntemin 10 adayı, başlıkta bulunan/toplam sayısı.
# 
# `grafik.png` — duyarlılık eğrisi: aşamalar, rastgele alt sınır ve 7 hekim.
# ----------------------------------------------------------------------

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

# --- sekil1_02.png: ornek roentgen -------------------------------------- #
dosya = "02.png"
g = oku(dosya)
e0 = kutular[(dosya, REFERANS)]
nk = adaylar(skor_a2(g, pencere, medyan), 10, secim["A3"]["mesafe"], dis_maskesi(g))

kullanildi = [False] * len(e0)
for x, y in nk:
    for i, (bx, by, bw_, bh) in enumerate(e0):
        if not kullanildi[i] and bx <= x <= bx+bw_ and by <= y <= by+bh:
            kullanildi[i] = True
            break

fig, ax = plt.subplots(figsize=(11, 8.7))
ax.imshow(g, cmap="gray", vmin=0, vmax=1)
for (x, y, w, h), var in zip(e0, kullanildi):
    ax.add_patch(Rectangle((x, y), w, h, fill=False, linewidth=2.2,
                           edgecolor="#f2c200" if var else "#e02020"))
for i, (x, y) in enumerate(nk, 1):
    ax.plot(x, y, marker="+", markersize=13, markeredgewidth=2.4, color="#2f7fe0")
    ax.annotate(str(i), (x, y), textcoords="offset points", xytext=(8, 5),
                fontsize=9, color="#2f7fe0", fontweight="bold")
ax.set_title(f"{dosya} - A3 yontemi, 10 aday | E0 kutusu: "
             f"{sum(kullanildi)} bulundu / {len(e0)} toplam")
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout(); fig.savefig("sekil1_02.png", dpi=150); plt.show()

# --- grafik.png: duyarlilik egrisi --------------------------------------- #
x = np.arange(1, K + 1)
renk = {"rastgele": "#8c8c8c", "A0": "#c6a700", "A1": "#e07b39",
        "A2": "#3b7dd8", "A3": "#1a7f37"}
fig, ax = plt.subplots(figsize=(7.5, 5.4))
for ad in ["A3", "A2", "A1", "A0", "rastgele"]:
    ax.plot(x, egri[ad], marker="o", markersize=3.5, linewidth=2, color=renk[ad],
            label=ad, linestyle="--" if ad == "rastgele" else "-")
for h, (duy, isa) in hekim.items():
    ax.scatter(isa, duy, s=60, marker="D", color="#b3306e", zorder=5)
    ax.annotate(h, (isa, duy), textcoords="offset points", xytext=(7, -3),
                fontsize=9, color="#b3306e")
ax.scatter([], [], s=60, marker="D", color="#b3306e", label="Hekimler")
ax.set_xlabel("Goruntu basina isaret sayisi")
ax.set_ylabel("Duyarlilik (E0 kutularinin bulunan orani)")
ax.set_xlim(0, K + 0.6); ax.set_ylim(0, 1.0); ax.grid(alpha=0.25); ax.legend(fontsize=9)
fig.tight_layout(); fig.savefig("grafik.png", dpi=160); plt.show()
