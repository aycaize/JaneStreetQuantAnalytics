# GRU Deney Logu — One-Step-Ahead (Sızıntısız)

Kaggle ortamında (GPU: T4/P100) çalıştırılır.
Aynı sızıntısız one-step-ahead mantığı: model her gün SADECE geçmişi görerek tahmin yapar.

**Sabit ayarlar (tüm deneylerde):**
- DATA_START: 700, DATA_END: 1699, TRAIN_CUTOFF: 1499 (Kaggle RAM'i yerelden daha kısıtlı olabilir, gerekirse 1000'e çekilir)
- Retrain frequency: her N günde bir (versiyon bazlı belirtilir)

---

## Faz 1: Feature Engineering (V1-V15)

| Versiyon | Değişiklik | hidden_size | num_layers | Utility | R² | Pos gün | Not |
|---|---|---|---|---|---|---|---|
| V1 | Baseline: 79 ham feature + NaN flag | 64 | 1 | — | — | — | İlk sızıntısız GRU referansı |
| V2 | hidden=128, layers=2 (önceki statik mimari) | 128 | 2 | — | — | — | |
| V3 | + Rolling mean w=20 (top4 feat) | 128 | 2 | — | — | — | |
| V4 | + lag1, lag5 | 128 | 2 | — | — | — | |
| V5 | Padding stratejisi: 968 sabit vs gerçek uzunluk | 128 | 2 | — | — | — | |
| V6 | Dropout 0.1 → 0.2 | 128 | 2 | — | — | — | |
| V7 | + Cross-sectional feature (rank/zscore) | 128 | 2 | — | — | — | |
| V8 | Sequence length: tüm gün (968) vs kısaltılmış | 128 | 2 | — | — | — | |
| V9 | Bidirectional GRU denemesi | 128 | 2 | — | — | — | |
| V10 | + Auxiliary target (responder_3) | 128 | 2 | — | — | — | |
| V11 | Embedding ekleme (symbol_id) | 128 | 2 | — | — | — | |
| V12 | Loss fonksiyonu: weighted MSE vs MAE | 128 | 2 | — | — | — | |
| V13 | Gradient clipping değeri testi (0.5 vs 1.0 vs 2.0) | 128 | 2 | — | — | — | |
| V14 | Online learning: her gün 1 update step ekle | 128 | 2 | — | — | — | Evgeniia yaklaşımı |
| V15 | En iyi V'lerin kombinasyonu | — | — | — | — | — | Faz 1 sonucu |

---

## Faz 2: Hyperparameter Tuning (V16-V30)

Sabit feature seti: **V15'in en iyi feature kombinasyonu**

| Versiyon | hidden_size | num_layers | lr | dropout | batch_size | Utility | R² | Not |
|---|---|---|---|---|---|---|---|---|
| V16 | 64 | 2 | 3e-4 | 0.1 | 4 | — | — | |
| V17 | 128 | 2 | 3e-4 | 0.1 | 4 | — | — | V15 ile aynı (referans) |
| V18 | 256 | 2 | 3e-4 | 0.1 | 4 | — | — | |
| V19 | 128 | 1 | 3e-4 | 0.1 | 4 | — | — | |
| V20 | 128 | 3 | 3e-4 | 0.1 | 4 | — | — | |
| V21 | 128 | 2 | 1e-4 | 0.1 | 4 | — | — | |
| V22 | 128 | 2 | 1e-3 | 0.1 | 4 | — | — | |
| V23 | 128 | 2 | 3e-4 | 0.2 | 4 | — | — | |
| V24 | 128 | 2 | 3e-4 | 0.3 | 4 | — | — | |
| V25 | 128 | 2 | 3e-4 | 0.1 | 8 | — | — | |
| V26 | 128 | 2 | 3e-4 | 0.1 | 16 | — | — | |
| V27 | (en iyi kombinasyon 1) | — | — | — | — | — | — | |
| V28 | (en iyi kombinasyon 2) | — | — | — | — | — | — | |
| V29 | Online learning lr testi (1e-4 vs 3e-4 vs 1e-3) | — | — | — | — | — | — | |
| V30 | Final: en iyi feature + en iyi hyperparam + online learning | — | — | — | — | — | — | **Final model** |

---

## Notlar

- GRU her deneyde epoch bazlı eğitim history'si de not edilmeli (kaç epoch'ta plateau).
- Kaggle session süresi dolarsa, model ağırlıklarını `/kaggle/working/` altına kaydet, sonraki session'da devam et.
- LightGBM Faz 1 sonuçlarıyla karşılaştırma yaparak hangi feature'ların GRU'da da işe yaradığını not et.
