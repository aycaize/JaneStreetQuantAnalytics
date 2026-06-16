# LightGBM Deney Logu — One-Step-Ahead (Sızıntısız)

Tüm deneyler aynı sızıntısız one-step-ahead pipeline ile test edilir:
- Rolling feature'lar her gün için SADECE o günden önceki window'dan hesaplanır
- Train/val ayrımı kesin: train günleri val rolling hesabına asla karışmaz
- Metrikler: utility score (Sharpe-benzeri) + weighted R²

**Sabit ayarlar (tüm deneylerde):**
- DATA_START: 1000, DATA_END: 1699, TRAIN_CUTOFF: 1499
- WINDOW_SIZE: 20 (gün) — aksi belirtilmedikçe

---

## Faz 1: Feature Engineering (V1-V15)

| Versiyon | Değişiklik | Feature sayısı | Utility | R² | Pos gün | Not |
|---|---|---|---|---|---|---|
| V1 | Baseline: 79 ham feature, NaN→0 | 79 | — | — | — | Sızıntısız ilk referans |
| V2 | + NaN flag (4 feature: 21,26,27,31) | 83 | — | — | — | |
| V3 | + Rolling mean w=20 (top4 feat) | 91 | — | — | — | |
| V4 | + Rolling mean+std w=20 (top4 feat) | 95 | — | — | — | İlk çalışan one-step-ahead (referans: 0.8049) |
| V5 | + lag1, lag5 (top4 feat) | 95+ | — | — | — | |
| V6 | Rolling window=10 (kısa) | — | — | — | — | |
| V7 | Rolling window=50 (uzun) | — | — | — | — | |
| V8 | top4 → top8 feature genişletme | — | — | — | — | |
| V9 | + Cross-sectional rank (top5) | — | — | — | — | |
| V10 | + Cross-sectional zscore (top5) | — | — | — | — | |
| V11 | + Lagged responder (günlük, sızıntısız) | — | — | — | — | |
| V12 | Rolling max/min ekle | — | — | — | — | |
| V13 | NaN flag + rolling kombinasyonu farklı top_feats | — | — | — | — | |
| V14 | Tüm feature'lar için rolling (79 feat × w=20) | — | — | — | — | Pahalı, memory testi |
| V15 | En iyi V'lerin kombinasyonu | — | — | — | — | Faz 1 sonucu |

---

## Faz 2: Hyperparameter Tuning (V16-V30)

Sabit feature seti: **V15'in en iyi feature kombinasyonu**

| Versiyon | num_leaves | learning_rate | min_child_samples | subsample | n_estimators | Utility | R² | Not |
|---|---|---|---|---|---|---|---|---|
| V16 | 64 | 0.05 | 50 | 0.8 | 300 | — | — | |
| V17 | 128 | 0.05 | 50 | 0.8 | 300 | — | — | V15 ile aynı (referans) |
| V18 | 256 | 0.05 | 50 | 0.8 | 300 | — | — | |
| V19 | 128 | 0.01 | 50 | 0.8 | 300 | — | — | |
| V20 | 128 | 0.1 | 50 | 0.8 | 300 | — | — | |
| V21 | 128 | 0.05 | 20 | 0.8 | 300 | — | — | |
| V22 | 128 | 0.05 | 100 | 0.8 | 300 | — | — | |
| V23 | 128 | 0.05 | 50 | 0.6 | 300 | — | — | |
| V24 | 128 | 0.05 | 50 | 1.0 | 300 | — | — | |
| V25 | 128 | 0.05 | 50 | 0.8 | 100 | — | — | |
| V26 | 128 | 0.05 | 50 | 0.8 | 500 | — | — | |
| V27 | (en iyi kombinasyon 1) | — | — | — | — | — | — | |
| V28 | (en iyi kombinasyon 2) | — | — | — | — | — | — | |
| V29 | (retrain frequency testi: her 10 gün) | — | — | — | — | — | — | |
| V30 | (final: en iyi feature + en iyi hyperparam) | — | — | — | — | — | — | **Final model** |

---

## Notlar

- Her versiyon çalıştırıldığında bu tabloyu güncelle.
- "Not" sütununa neden bu sonucun çıktığına dair kısa yorum ekle (örn. "window küçüldükçe gürültü arttı").
- En iyi versiyon her fazın sonunda **kalın** ile işaretlenecek.
