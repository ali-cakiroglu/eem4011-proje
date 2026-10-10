"""Bitewing röntgenlerde klasik görüntü işlemeyle çürük adayı işaretleme.

Veri: "Dental caries in bitewing radiographs" (Mendeley Data, CC BY-NC 3.0)
      100 röntgen (1068x847 PNG). 8 işaretleyicinin kutuları COCO JSON'da ayrı
      birer "category": E0 (referans uzman), E1-E4 (uzman), N1-N3 (az deneyimli).

Yöntem dört aşamalı ve hiçbir aşaması veriden öğrenmiyor:
    A0  ham koyuluk
    A1  yerel arka plan çıkarma  (H3 taban çizgisi + H7 ortalama filtre)
    A2  medyan filtre            (H7)
    A3  Otsu eşiğiyle diş içi kısıtı (H7)

Ortalama filtre, medyan filtre ve Otsu eşiği burada numpy ile elle yazıldı;
scipy.ndimage.uniform_filter / median_filter ve skimage.filters.threshold_otsu
ile aynı işi yapıyorlar, ek kurulum gerektirmiyorlar.
"""

import json
from pathlib import Path

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image

KOK = Path(__file__).resolve().parent.parent / "data" / "bitewing" / "dental_rtg_test"
GORUNTU = KOK / "images"
ETIKET = KOK / "test_annotations_anonymized.json"

REFERANS = "E0"
HEKIMLER = ["E1", "E2", "E3", "E4", "N1", "N2", "N3"]

# Parametreler yalnız bu bölümde seçilir, sonuç öbür bölümde bir kez ölçülür.
PARAMETRE = [f"{i:02d}.png" for i in range(1, 21)]      # 01-20.png
SONUC = [f"{i:02d}.png" for i in range(21, 101)]        # 21-100.png


# --------------------------------------------------------------------------- #
# Veri
# --------------------------------------------------------------------------- #

def yukle_etiketler() -> dict[tuple[str, str], list[tuple[float, float, float, float]]]:
    """(dosya adı, işaretleyici) -> [(x, y, genişlik, yükseklik), ...] sözlüğü."""
    d = json.loads(ETIKET.read_text(encoding="utf-8"))
    isaretleyici = {c["id"]: c["name"] for c in d["categories"]}
    dosya = {im["id"]: im["file_name"] for im in d["images"]}
    kutular: dict[tuple[str, str], list] = {}
    for a in d["annotations"]:
        anahtar = (dosya[a["image_id"]], isaretleyici[a["category_id"]])
        kutular.setdefault(anahtar, []).append(tuple(float(v) for v in a["bbox"]))
    return kutular


def oku(dosya: str) -> np.ndarray:
    """Röntgeni [0, 1] aralığında gri tonlamalı diziye çevirir (yüksek = parlak)."""
    return np.asarray(Image.open(GORUNTU / dosya).convert("L"), dtype=np.float32) / 255.0


# --------------------------------------------------------------------------- #
# Temel işlemler (H7)
# --------------------------------------------------------------------------- #

def ortalama_filtre(g: np.ndarray, k: int) -> np.ndarray:
    """k x k ortalama filtre. Integral görüntüyle hesaplanır, pencere boyutundan
    bağımsız olarak aynı sürede çalışır."""
    r = k // 2
    p = np.pad(g, r, mode="reflect").astype(np.float64)
    bt = np.pad(p.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    H, W = g.shape
    toplam = (bt[k:k + H, k:k + W] - bt[0:H, k:k + W]
              - bt[k:k + H, 0:W] + bt[0:H, 0:W])
    return (toplam / (k * k)).astype(np.float32)


def medyan_filtre(g: np.ndarray, k: int) -> np.ndarray:
    """k x k medyan filtre. Aykırı pikselleri atar, kenarı ortalama kadar bulandırmaz."""
    r = k // 2
    p = np.pad(g, r, mode="reflect")
    return np.median(sliding_window_view(p, (k, k)), axis=(-2, -1)).astype(np.float32)


def otsu_esigi(g: np.ndarray, kutu: int = 256) -> float:
    """Otsu (1979): gri seviye histogramında sınıflar arası varyansı en çok yapan eşik."""
    h, kenar = np.histogram(g, bins=kutu, range=(0.0, 1.0))
    h = h.astype(np.float64)
    merkez = (kenar[:-1] + kenar[1:]) / 2.0
    agirlik0 = h.cumsum()
    agirlik1 = h.sum() - agirlik0
    kumulatif = (h * merkez).cumsum()
    ortalama0 = np.divide(kumulatif, agirlik0, out=np.zeros_like(kumulatif),
                          where=agirlik0 > 0)
    ortalama1 = np.divide(kumulatif[-1] - kumulatif, agirlik1,
                          out=np.zeros_like(kumulatif), where=agirlik1 > 0)
    arasi = agirlik0 * agirlik1 * (ortalama0 - ortalama1) ** 2
    arasi[(agirlik0 == 0) | (agirlik1 == 0)] = 0.0
    return float(merkez[int(np.argmax(arasi))])


def dis_maskesi(g: np.ndarray) -> np.ndarray:
    """Otsu eşiğinin üstünde kalan parlak bölge: diş. Altı yumuşak doku, hava ve
    siyah üst-alt kenarlar."""
    return g > otsu_esigi(g)


# --------------------------------------------------------------------------- #
# Aşamaların skor haritaları
# --------------------------------------------------------------------------- #

def skor_a0(g: np.ndarray) -> np.ndarray:
    """Ham koyuluk: lezyon çevresinden daha az ışık geçirdiği için daha koyudur."""
    return 1.0 - g


def skor_a1(g: np.ndarray, pencere: int) -> np.ndarray:
    """Yerel arka plan çıkarma: yavaş değişen bileşen (pozlama, diş kalınlığı)
    büyük pencereli ortalamayla tahmin edilip çıkarılır. Skor artık mutlak
    koyuluk değil, 'çevresine göre ne kadar koyu'."""
    return ortalama_filtre(g, pencere) - g


def skor_a2(g: np.ndarray, pencere: int, medyan: int) -> np.ndarray:
    """A1 + medyan filtre: fark görüntüsündeki tek piksellik sıçramalar kendi
    başına tepe üretip aday bütçesini harcıyor, medyan bunları bastırır."""
    return medyan_filtre(skor_a1(g, pencere), medyan)


# --------------------------------------------------------------------------- #
# Aday seçimi
# --------------------------------------------------------------------------- #

def adaylar(skor: np.ndarray, n: int, en_az_mesafe: int,
            maske: np.ndarray | None = None) -> list[tuple[int, int]]:
    """Skor haritasından n aday: en yüksek tepeyi al, çevresini kapat, tekrarla.
    Adaylar skor sırasına göre döner; en_az_mesafe iki adayın birbirine ne kadar
    yaklaşabileceğini sınırlar."""
    s = skor.astype(np.float32, copy=True)
    if maske is not None:
        s[~maske] = -np.inf
    H, W = s.shape
    r = int(en_az_mesafe)
    secilen: list[tuple[int, int]] = []
    for _ in range(n):
        düz = int(np.argmax(s))
        y, x = divmod(düz, W)
        if not np.isfinite(s[y, x]):
            break
        secilen.append((x, y))
        s[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = -np.inf
    return secilen


def rastgele_adaylar(maske: np.ndarray, n: int, uret: np.random.Generator
                     ) -> list[tuple[int, int]]:
    """Alt sınır: diş maskesi içinden rastgele n nokta."""
    ys, xs = np.nonzero(maske)
    if len(ys) == 0:
        return []
    secim = uret.choice(len(ys), size=min(n, len(ys)), replace=False)
    return [(int(xs[i]), int(ys[i])) for i in secim]


# --------------------------------------------------------------------------- #
# Ölçüm
# --------------------------------------------------------------------------- #

def profil(noktalar: list[tuple[int, int]],
           kutular: list[tuple[float, float, float, float]],
           k_azami: int) -> np.ndarray:
    """Merkez-kutu kuralı: bir işaretin merkezi bir E0 kutusunun içindeyse o kutu
    bulunmuş sayılır. Her kutu en fazla bir kez sayılır. Dönen dizi, ilk k işaret
    kullanıldığında bulunan kutu sayısıdır (k = 1..k_azami).

    Not: makale simetrik kural kullanıyor ('birinin merkezi diğerinin içinde ya da
    tersi'); adaylar nokta olduğu için burada kuralın tek yönü uygulanıyor ve aynı
    kural karşılaştırılan her şeye (yöntem, rastgele, hekimler) aynen uygulanıyor.
    """
    kullanildi = [False] * len(kutular)
    cikti = np.zeros(k_azami, dtype=np.int32)
    bulunan = 0
    for k in range(k_azami):
        if k < len(noktalar):
            x, y = noktalar[k]
            for i, (bx, by, bw, bh) in enumerate(kutular):
                if not kullanildi[i] and bx <= x <= bx + bw and by <= y <= by + bh:
                    kullanildi[i] = True
                    bulunan += 1
                    break
        cikti[k] = bulunan
    return cikti


def hekim_duyarliligi(kutular: dict, bolum: list[str], hekim: str) -> tuple[float, float]:
    """Bir hekimin E0'a göre duyarlılığı ve görüntü başına kutu sayısı.
    Hekimin kutuları merkezlerine indirgenip aynı merkez-kutu kuralıyla eşleştirilir."""
    bulunan = hedef = isaret = 0
    for dosya in bolum:
        e0 = kutular.get((dosya, REFERANS), [])
        kendi = kutular.get((dosya, hekim), [])
        merkezler = [(x + w / 2, y + h / 2) for x, y, w, h in kendi]
        hedef += len(e0)
        isaret += len(merkezler)
        if e0 and merkezler:
            bulunan += int(profil(merkezler, e0, len(merkezler))[-1])
    return (bulunan / hedef if hedef else 0.0,
            isaret / len(bolum) if bolum else 0.0)
