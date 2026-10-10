# Bitewing röntgende klasik yöntemle çürük adayı işaretleme

Açık bir bitewing röntgen veri setinde, eğitim gerektirmeyen klasik görüntü işleme adımlarıyla her görüntüde 10 çürük adayı işaretleyip uzman E0'ın kutularına göre Duyarlılık@10'u ölçen bir çalışma.

## Veri

**Dental caries in bitewing radiographs** — Mendeley Data, V1 (Kybic, Tichý, Kunt; 2023)
https://data.mendeley.com/datasets/4fbdxs7s7w/1 · DOI: 10.17632/4fbdxs7s7w.1 · **Lisans: CC BY-NC 3.0**

100 bitewing röntgen (PNG, 1068×847) ve 8 işaretleyicinin kutuları COCO JSON biçiminde: E0 (referans uzman), E1–E4 (uzman), N1–N3 (az deneyimli).

**Veri bu depoda yoktur.** Defter çalışırken Mendeley'den iner (~75 MB) ve SHA-256 özetiyle doğrulanır. Veri kimliksizleştirilmiştir, kişisel veri içermez; bu depoda hiçbir hasta verisi veya görüntü tutulmamaktadır.

## Colab'da çalıştırma

1. `bitewing_colab.ipynb` dosyasını indir.
2. [colab.research.google.com](https://colab.research.google.com) → **Dosya → Not defteri yükle** → dosyayı seç.
3. **Çalışma zamanı → Tümünü çalıştır.**
4. İlk hücre veriyi indirir ve SHA-256 özetini doğrular; "SHA-256 dogru" yazmalı.
5. Hücreler sırayla: temel işlemler → veri okuma → aşamalar → parametre seçimi (01–20) → sonuç (21–100) → hekimler → şekiller.

Ek kurulum gerekmez: yalnız `numpy`, `Pillow` ve `matplotlib` kullanılır, üçü de Colab'da hazır gelir. Toplam süre yaklaşık 5 dakika.

## Yöntem

Dört aşama, hiçbiri veriden öğrenmez:

| Aşama | Ne ekliyor |
|---|---|
| A0 | ham koyuluk |
| A1 | + yerel arka plan çıkarma (ortalama filtre) |
| A2 | + medyan filtre |
| A3 | + Otsu eşiğiyle diş içi kısıtı |

Her aşamanın skor haritasından görüntü başına 10 aday, en yüksek tepeyi alıp çevresini kapatarak seçilir. Bir işaretin merkezi bir E0 kutusunun içine düşüyorsa o kutu bulunmuş sayılır (merkez-kutu kuralı); her kutu en fazla bir kez sayılır. Aynı kural yönteme, rastgele işaretlemeye ve hekimlere birebir uygulanır.

Parametreler (ortalama filtre penceresi, medyan penceresi, adaylar arası en az mesafe) yalnız `01–20.png` üzerinde seçilir; `21–100.png` bu seçimden sonra bir kez ölçülür.

## Şekiller

### `sekil1_02.png`

Örnek röntgen (02.png). Sarı kutu: E0'ın işaretlediği ve yöntemin bulduğu lezyon. Kırmızı kutu: E0'ın işaretlediği ama yöntemin kaçırdığı lezyon. Mavi `+`: yöntemin seçtiği 10 aday, seçilme sırasıyla numaralı. Başlıktaki sayı: bu görüntüde kaç E0 kutusunun bulunduğu / toplam kaç E0 kutusu olduğu.

> **Veri:** Kybic, Tichý, Kunt, Mendeley Data, DOI: 10.17632/4fbdxs7s7w.1, CC BY-NC 3.0.

### `grafik.png`

Duyarlılık–aday sayısı eğrisi. Dört aşama ve rastgele alt sınır eğri olarak; 7 hekim, kendi ortalama işaret sayılarında birer nokta olarak.

> **Veri:** Kybic, Tichý, Kunt, Mendeley Data, DOI: 10.17632/4fbdxs7s7w.1, CC BY-NC 3.0.

## Ana sonuç

`21–100.png` üzerinde (62 görüntü, 218 E0 kutusu), görüntü başına 10 aday:

| Aşama | Duyarlılık@10 |
|---|---|
| Rastgele işaretleme (alt sınır) | 0,055 |
| A0 ham koyuluk | 0,000 |
| A1 + yerel arka plan çıkarma | 0,046 |
| A2 + medyan filtre | 0,092 |
| **A3 + Otsu diş içi kısıtı** | **0,174** |

A3, rastgele işaretlemenin **3,2 katı**. Aynı kuralla ölçüldüğünde 7 hekim, görüntü başına 2,5–4,3 işaretle 0,74–0,91 duyarlılık alıyor; yöntem bu düzeye ancak görüntü başına 150–200 adayla ulaşıyor.

## Sağlamalar

Defter çalıştığında şu üç sayı tutmalı; tutmuyorsa veri ya da okuma hatalıdır:

| Sağlama | Beklenen |
|---|---|
| E0'ın toplam kutu sayısı | **269** |
| E0'ın kutu sayısı, test bölümü (21–100.png) | **218** |
| Kutu merkezlerinin geçerlilik oranı | **1,000** |

Son satır, her E0 kutusunun merkezinin hem görüntü sınırları içinde hem kendi kutusunun içinde kaldığını doğrular — merkez-kutu kuralının anlamlı olması buna bağlı.

Ayrıca veri `01–100.png` adıyla, 1068×847 boyutunda ve sekiz işaretleyiciyle (E0, E1–E4, N1–N3, toplam 2.585 kutu) gelmelidir; indirilen zip 79.048.142 bayttır ve SHA-256 özeti defterde doğrulanır.

## Uygulanabilirlik

- **Eğitim verisi gerekmez.** Yöntemin hiçbir aşaması veriden öğrenmez; ağırlık, model dosyası ya da etiketli eğitim kümesi yoktur. Etiketler yalnız ölçüm için kullanılır.
- **GPU gerekmez.** Tamamı CPU'da çalışır.
- **Görüntü başına süre:** CPU'da ortalama **226 ms** (medyan 168 ms, aralık 149–379 ms; A3 boru hattı + 10 aday seçimi, görüntü okuma hariç). 100 görüntü yaklaşık 23 saniye.
- Üç parametre (ortalama filtre penceresi, medyan penceresi, adaylar arası en az mesafe) dışında ayar yoktur.

## Kaynaklar

1. Kunt L., Kybic J. "Automatic caries detection in bitewing radiographs: part I — deep learning." *Clinical Oral Investigations*, 2023. DOI: [10.1007/s00784-023-05335-1](https://doi.org/10.1007/s00784-023-05335-1)
2. Tichý A., Kunt L., Nagyová V., Kybic J. "Automatic caries detection in bitewing radiographs — Part II: experimental comparison." *Clinical Oral Investigations*, 2024, 28(2):133. DOI: [10.1007/s00784-024-05528-2](https://doi.org/10.1007/s00784-024-05528-2)

Veri setinin E0 / E1–E4 / N1–N3 adlandırması, işaretleyicilerin deneyim düzeyleri ve merkez-kutu eşleştirme kuralı 2 numaralı kaynaktan alınmıştır.

## Lisans

- **Kod:** MIT (bkz. `LICENSE`)
- **Veri:** CC BY-NC 3.0 — veri bu depoda yoktur, yalnız bağlantısı verilmiştir; ticari olmayan kullanım.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `bitewing_colab.ipynb` | Colab defteri (baştan sona çalışır) |
| `bitewing_colab.py` | Defterin düz Python hali |
| `bitewing.py` | Modül: veri okuma, temel işlemler, aşamalar, ölçüm |
| `calistir.py` | Parametre taraması ve nihai ölçüm |
| `grafik.py`, `sekil.py` | Şekilleri üreten betikler |
| `sonuclar.json` | Ölçülen bütün sayılar |
| `LICENSE` | MIT kod lisansı |
