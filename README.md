# Jane Street Real-Time Market Data Forecasting

Kaggle Jane Street yarışmasında sistematik bir ML pipeline geliştirme süreci.
Amaç: finansal zaman serisi tahmini için EDA'dan online learning'e kadar tam bir pipeline.

**Yarışma:** [Jane Street Real-Time Market Data Forecasting](https://www.kaggle.com/competitions/jane-street-real-time-market-data-forecasting)  
**Metrik:** Weighted zero-mean R² (+ utility score takibi)  
**Veri:** 47M satır, 1699 gün, 39 symbol, 79 feature

---

## Sonuçlar

| Model | Utility | R² | Notlar |
|---|---|---|---|
| LightGBM baseline | 1.0227 | 0.0136 | 79 ham feature |
| + NaN flag | 1.0820 | 0.0148 | +5.9% — yapısal kırılma sinyali |
| + Rolling top5 w=500 | 1.0899 | 0.0150 | Symbol bazlı rolling |
| Versiyon A — rolling top10 w=500 | 1.0999 | 0.0152 | Genişletilmiş rolling |
| Versiyon B — rmean100+lag1+lag5 | 1.1277 | 0.0152 | LightGBM best |
| GRU (hidden=128, 2 layer, 30 epoch) | 1.1847 | — | Zamansal dinamikler |
| LightGBM one-step-ahead (devam ediyor) | TBD | — | Sliding window CV |

---

## Proje Yapısı

```
├── notebooks/
│   ├── 01_eda.ipynb             EDA: veri yapısı, NaN analizi, volatilite rejimleri
│   ├── 02_lgbm_experiments.ipynb LightGBM deneyleri ve feature engineering
│   ├── 03_gru_model.ipynb       GRU mimarisi ve eğitim
│   └── 04_online_learning.ipynb One-step-ahead sliding window CV
├── src/
│   ├── features.py              Feature engineering fonksiyonları
│   ├── models.py                GRU mimarisi (PyTorch)
│   ├── train.py                 Eğitim pipeline
│   └── evaluate.py              Utility score, weighted R²
└── results/
    └── experiment_log.md        Tüm deney sonuçları
```

---

## Yaklaşım

### 1. EDA Bulguları

**Veri yapısı:**
- 47M satır: 1699 gün × 968 time_id × 39 symbol
- 10 partition × 170 gün (eşit bölünmüş — CV için doğal sınırlar)
- 3 symbol geç girmiş: symbol 6, 18, 32 (date_id=1063'ten itibaren)

**Kritik NaN pattern:**
```
feature_21, 26, 27, 31:
  Partition 0-2 (date_id 0-509): %100 NaN
  Partition 3   (date_id 510-679): %10 NaN (geçiş)
  Partition 4+  (date_id 680+): ~%0 NaN

→ Veri kaynağı date_id ~510-680'de değişmiş
→ date_id=700'den başlamak bu yüzden kritik
```

**Responder korelasyonları:**
```
responder_6 ↔ responder_3: 0.727 (en yüksek)
responder_6 ↔ responder_8: 0.447
responder_6 ↔ responder_7: 0.432
responder_6 ↔ responder_0: -0.120 (negatif)
```

**Volatilite rejimleri:**
```
Sakin dönemler: std ≈ 0.45 (date_id 36, 1245-1255)
Kriz dönemleri: std ≈ 2.35 (date_id 786-790, küme halinde)
→ Online learning bu geçişler için şart
```

### 2. CV Stratejisi

```
Train:      date_id 700-1359 (partition 4-7)
Gap:        10 gün (sızıntı önleme)
Validation: date_id 1370-1529 (partition 8)

One-step-ahead sliding window:
  Her gün ayrı tahmin → gerçek değerler gelince güncelle
  Sliding window boyutu: 660 gün
```

### 3. Feature Engineering

**En etkili feature'lar (importance sırasına göre):**
```
feature_01_rmean100  (symbol bazlı 100-adım rolling mean)
feature_08_rmean100
feature_60_rmean100
feature_03_rmean100
feature_07          (ham feature, güçlü sinyal)
```

**Denenip reddedilenler:**
```
✗ Gecikmeli market mean     → gürültü, utility -0.088
✗ Short rolling w=10        → rmean100 ile çakışıyor
✗ Günlük lagged responder   → zayıf sinyal (çok kaba)
✗ Time_id bazlı lagged resp → DATA LEAKAGE (sızıntı)
```

**Data leakage tespiti:**
```
Hatalı: shift(1).over('symbol_id') ile responder lag
  → time_id=500 için time_id=499'un responder'ını verdi
  → O an bilinmeyen bilgi → utility 1.08 → 3.69 (sahte)

Doğru: günlük ortalama → shift(1) (dün bilgisi)
  → Utility 1.0998 (gerçekçi)
```

### 4. Model Mimarisi

**LightGBM:**
```python
LGBMRegressor(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=128,
    min_child_samples=50,
    subsample=0.8,
    colsample_bytree=0.8
)
```

**GRU (PyTorch):**
```python
# Seq2Seq: her time_id için ayrı tahmin
# Input:  (batch, 968, 83)  # 968 time adımı, 83 feature
# Output: (batch, 968)

GRU(input=83, hidden=128, layers=2, batch_first=True)
→ Linear(128, 64) → ReLU → Dropout(0.1) → Linear(64, 1)

# Padding: eksik time_id'ler sıfırla dolduruldu
# Loss: weighted MSE (sadece gerçek satırlarda)
```

### 5. Online Learning

```
Her gün:
1. Mevcut model ile o günü tahmin et
2. Sliding window (son 660 gün) ile modeli yeniden eğit
3. Eski modeli yeni modelle değiştir

Beklenen katkı: +0.05 ile +0.15 utility
```

---

## Öğrenilen Dersler

**Feature engineering:**
- NaN binary flag → yapısal bilgi taşıyor (+5.9%)
- Symbol bazlı rolling mean (w=100) > kısa window (w=10)
- Market geneli değil, symbol özgü istatistikler daha güçlü
- Data leakage: time_id bazlı shift vs gün bazlı shift farkı kritik

**Model seçimi:**
- GRU zamansal dinamikleri LightGBM'den iyi öğreniyor
- Transformer: cross-sectional bağ güçlüyse ekle, değilse ekleme
- Basit model + doğru feature > karmaşık model + yanlış feature

**Validation:**
- Tek fold utility oynak (160 gün küçük sample)
- One-step-ahead sliding window gerçek inference'ı simüle ediyor
- Data leakage kontrolü: R² ve utility birlikte takip et
  - R² beklenmedik artış → sızıntı şüphesi
  - utility 3.69, R² 0.87 → kesin sızıntı

---

## Referanslar

- [8. sıra çözümü (Evgeniia Grigoreva)](https://github.com/evgeniavolkova/kagglejanestreet)
- hydantess: Optiver 1., Enefit 1., Jane Street 5. — online learning + GRU imzası
- [Minsky Financial Instability Hypothesis](https://www.investopedia.com/terms/m/minsky-moment.asp) — rejim değişimi motivasyonu

---

## Kurulum

```bash
pip install polars lightgbm torch kaggle
```

```python
# Veri indirme
import kaggle
kaggle.api.competition_download_files(
    'jane-street-real-time-market-data-forecasting'
)
```
