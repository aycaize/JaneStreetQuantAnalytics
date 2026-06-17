# Faz 3: Online Learning (V31+) — LightGBM

**Önemli not (mevcut V1-V30 sonuçlarının yorumlanması için):**

V1-V30 boyunca model SADECE BİR KEZ eğitiliyor (train_cutoff'a kadar).
Validation aşamasında her gün için:
  - Feature'lar (rolling mean/std/lag) o günden önceki son `window_size` günden
    TAZE hesaplanıyor (sliding window, sızıntısız)
  - AMA model katsayıları (ağaç yapıları) 199 validation günü boyunca SABİT kalıyor

Yani V1-V30 sonuçları "sabit model + taze input feature" performansını ölçüyor.
Bu BİLİNÇLİ bir tasarım kararı: feature engineering'in kendi başına katkısını
online learning etkisiyle karıştırmadan izole ölçmek için.

Gerçek competition senaryosunda (ve top çözümlerde - Evgeniia, hydantess) model de
periyodik olarak güncelleniyor. Bu etkiyi Faz 3'te ayrı ölçeceğiz:

| Versiyon | Açıklama | Retrain sıklığı | Utility | R² | Not |
|---|---|---|---|---|---|
| V31 | En iyi V1-V30 feature+hyperparam + online learning | Her 10 gün | — | — | — |
| V32 | Aynı + farklı retrain sıklığı | Her 20 gün | — | — | — |
| V33 | Aynı + farklı retrain sıklığı | Her 5 gün | — | — | Retrain sıklığı/maliyet trade-off |

**Beklenti:** Online learning eklenince utility'nin V1-V30'daki en iyi statik
sonuçtan daha yüksek çıkması (çünkü model piyasa rejim değişimlerine adapte olabilecek).
