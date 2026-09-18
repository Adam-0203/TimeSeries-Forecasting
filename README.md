# Time Series Forecasting: A Comparative Study of Refinement Techniques

> Benchmarking ARIMA, Fourier, Box-Cox, and differencing methods across six IoT time series datasets — then going one step further with a novel Bayesian method that incorporates exogenous variables to refine forecasts.

---

## Overview

This repository contains a systematic comparison of forecasting refinement techniques applied to six real-world IoT time series datasets. The goal is simple: **determine which method works best depending on the data's characteristics** (seasonality, variance, trend, sample size).

All models are evaluated using **MAPE** (Mean Absolute Percentage Error) for scale-independent comparison.

---

## Datasets

| Dataset | Observations | Range | Seasonality | Trend | Timestamp |
|---------|-------------|-------|-------------|-------|-----------|
| Electro-Cardio-Gramme | 1,500 | [422, 684] | 722 | None | ms |
| Energy Consumption | 3,000 | [19,247, 44,537] | 18 | None | hour |
| Morocco Internet | 23 | [0.69, 89.9] | None | Increasing | year |
| Mexico Weather | 9,000 | [34, 97] | 7,201 | None | hour |
| Sensor | 5,000 | [6.88, 149.6] | None | None | hour |
| Temperature | 7,056 | [5.35, 36.5] | 24 | None | hour |

All datasets sourced from [Kaggle](https://www.kaggle.com/).

---

## Methods Compared

1. **Mean Baseline** — simple average as reference
2. **IQR Outlier Replacement** — remove outliers using Interquartile Range
3. **ARIMA** — Autoregressive Integrated Moving Average
4. **Fourier Series** — trigonometric approximation for seasonality
5. **Box-Cox Transformation** — variance stabilization
6. **Seasonal Differencing** — remove seasonality at seasonal lag
7. **First-Order Differencing** — remove trend
8. **LSTM** — deep learning baseline (Temperature dataset only)

---

## Results (MAPE %)

| Dataset | Mean | IQR | ARIMA | Fourier | Box-Cox | Seasonal Diff | 1st-Order Diff | **Best** |
|---------|------|-----|-------|---------|---------|---------------|----------------|----------|
| ECG | 4.50 | 4.47 | 4.47 | 4.56 | 4.39 | **7.75** | – | **Seasonal Diff** |
| Energy | 16.47 | 11.16 | 11.16 | 18.27 | 11.53 | **13.00** | – | **Seasonal Diff** |
| Morocco Internet | 59.93 | 56.68 | 40.72 | 55.92 | 28.87 | – | **12.74** | **1st-Order Diff** |
| Mexico Weather | 23.29 | 10.35 | **10.35** | 21.41 | 10.94 | 13.70 | – | **ARIMA / IQR** |
| Sensor | 11.15 | **4.45** | 10.18 | 10.54 | 10.30 | 12.15 | – | **IQR** |
| Temperature | 24.10 | 23.51 | 22.00 | 23.51 | **21.33** | 56.74 | – | **Box-Cox** |

---

* LSTM achieved the lowest MSAE (14.22) on Temperature, outperforming all classical methods on this high-frequency data.
* Despite Box-Cox having the best MAPE, seasonal differencing captured the ECG pattern best. Its higher MAPE is partly due to a slight prediction shift, highlighting the importance of visualizing predictions alongside metrics. The same remark can be made about the Energy consumption dataset.
* For rapidly fluctuating, high-frequency data (e.g., Temperature), no classical method consistently outperformed the others. In such cases, LSTM is a reasonable choice.

---

## Key Takeaways

| Data Characteristic | Recommended Approach |
|---------------------|---------------------|
| Large seasonality | Fourier series / Seasonal differencing |
| Clear pattern or trend | First-order differencing |
| Large variance | Box-Cox transformation |
| Small dataset | Trend extraction & differencing |
| High-frequency timestamp | Aggregate to appropriate lags |
| Outliers | IQR outlier replacement |

### Critical Insights

- **No universal best method** — performance depends entirely on data characteristics
- **Visual inspection matters** — the lowest MAPE doesn't always mean the best forecast (see ECG: seasonal differencing had the worst MAPE but captured the underlying behavior best)
- **Small datasets** require different strategies entirely (Morocco Internet: 23 observations)
- **LSTM shows promise** for complex, high-frequency data where classical methods fail

---

# Part 2 — Original Method: Bayesian Adjustment with Exogenous Variables

## Motivation

Classical time series models (ARIMA, Fourier, SARIMA) rely **solely on the history of the target variable**. They ignore auxiliary information that may influence future values — even when such information is readily available in IoT environments:

- **Humidity** → affects temperature
- **Traffic** → affects energy consumption
- **Day of week** → affects sensor readings

The core idea: **if we can predict the *direction* of the next change with confidence, we can correct the base forecast accordingly.**

## Logic Behind the Method

The method combines two components:

1. **Base model (ARIMA):** captures the temporal dynamics and produces a point forecast $\hat{y}_t^{\text{ARIMA}}$.
2. **Bayesian classifier:** uses the last 20 lagged values of an exogenous variable $X$ to predict whether the target will go **up or down** at the next step:

$$
\mathbf{x}_t = [x_{t-1}, x_{t-2}, \ldots, x_{t-20}], \qquad
Y_t = \begin{cases} 1 & \text{if } y_t > y_{t-1} \\ 0 & \text{otherwise} \end{cases}
$$

The classifier outputs a **probability** $p_t = P(Y_t = 1 \mid \mathbf{x}_t)$, which is then used to adjust the base forecast:

$$
\hat{y}_t^{\text{final}} = \hat{y}_t^{\text{ARIMA}} \times \left(1 + \alpha \cdot (p_t - 0.5)\right)
$$

- $p_t > 0.5$ → forecast is increased (upward move likely)
- $p_t < 0.5$ → forecast is decreased
- $p_t = 0.5$ → no adjustment (maximum uncertainty)

## Why Bayesian?

A classical classifier (e.g., logistic regression) would give us a hard **0/1 decision** or a point probability. The Bayesian approach gives us something fundamentally richer:

- **A full posterior distribution** over the classifier weights $\mathbf{w}$
- **A predictive distribution** for $P(Y_t = 1)$ — not just a point estimate
- **Quantified confidence** in each prediction, which can be used to *shrink* the adjustment when the model is uncertain (via the posterior variance of $\mathbf{w}$)

This is what enables **graded adjustments** rather than abrupt ones: instead of "flip the forecast up or down," the Bayesian output tells us *how strongly* to adjust. This is the key value added by choosing a Bayesian model over a deterministic classifier.

## Why These Weight Functions?

Two weight functions were tested to translate the probability $p_t$ into a correction magnitude:

| Weight function | Behavior |
|-----------------|----------|
| $1 + \alpha \cdot (p_t - 0.5)$ | Linear: symmetric and bounded; adjustment scales linearly with confidence. |
| $\log(p_t + e - 0.5)$ | Logarithmic: compresses high-confidence adjustments and emphasizes mid-range probabilities; more conservative near the extremes. |

The **logarithmic weight** produced the best results on the Mexico Weather dataset because it **avoids over-adjusting** when the classifier is very confident — a crucial safeguard since the Bayesian classifier is not infallible. The linear weight was kept as a baseline to confirm the log version's advantage was not simply a matter of scale.

## Results

Evaluated on the **Mexico Weather** dataset (temperature as target, humidity as exogenous variable):

| Method | MAPE |
|--------|------|
| ARIMA (baseline) | 11.21% |
| **ARIMA + Bayesian adjustment** | **11.13%** |

The improvement is **modest but consistent**, and — critically — it is **interpretable**: the Bayesian adjustment helps most during **sudden humidity changes**, where the base ARIMA model lags behind. Looking at the forecast plots, the improvement is particularly visible in the regions where ARIMA overshot the actual values.

## When Does It Help Most?

Based on a confusion-matrix analysis of the classifier, the method performs best when:

- The exogenous variable is **causally or strongly correlated** with the target's direction of change
- The base model struggles specifically during **regime changes** (sudden shifts)
- We can afford to be **selective** — applying the adjustment only when the classifier is reliably correct in one direction

## Future Extensions

- Replace ARIMA with a more flexible base model (**LSTM**, **Gradient Boosting**)
- Extend from binary (up/down) to **multi-class** direction prediction
- Apply to **UAV networks** and **wireless communications** where exogenous signals are abundant

---

## Requirements

pandas
numpy
matplotlib
statsmodels
scipy
scikit-learn
pymc # for Bayesian logistic regression
tensorflow # for LSTM


---

## Author

**Adam Hajjaji** — UM6P, College of Computing
Supervisor: Pr. Hajar EL HAMMOUTI

---

## License

MIT
