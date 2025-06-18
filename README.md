````markdown
# 🌊 Wave Farm Energy Prediction with Neural Networks

This project applies machine learning to predict the total power output of wave energy converter (WEC) farms based solely on buoy layout configuration. Using simulation-based datasets from Perth and Sydney, we trained a neural network to model the relationship between spatial layout and energy generation — with implications for real-world coastal planning, including the Philippines.

---

## 📌 Project Overview

- **Objective**: Predict the total energy output of a wave farm from its layout using a neural network.
- **Model Used**: `MLPRegressor` from scikit-learn.
- **Key Dataset**: `WEC_Perth_49.csv` — 49-buoy layouts with 36,043 samples.
- **SDG Focus**: United Nations Sustainable Development Goal 7 (Affordable and Clean Energy).

---

## ⚙️ Features

- 📊 Trains a neural network to learn from buoy position data (X1–Y49).
- 🔍 Evaluates performance using MSE, MAE, and Percent Error.
- 🌏 Tests generalization on a Sydney dataset.
- 🇵🇭 Applies the trained model to a sample Philippine layout for feasibility testing.
- 🔁 Compares activation functions (`relu`, `tanh`, `logistic`).
- 📈 Visualizes predictions vs actual outputs.

---

## 🧠 Key Results

| Dataset       | MAE (W) | Percent Error |
|---------------|---------|----------------|
| **Perth Test Set** | ~25,062 | ~0.63% ✅ |
| **Sydney Layout** | ~166,534 | ~4.14% ❗ |
| **PH Sample Layout** | Predicted Output: ~5.35 MW |

- `ReLU` was the best-performing activation.
- Model generalizes reasonably but is sensitive to wave climate differences.

---

## 📁 Project Structure

```bash
.
├── wec-mlp-model.py            # Main notebook/script
├── sample_philippine_layout.csv
├── WEC_Perth_49.csv
├── WEC_Sydney_49.csv
├── figures/                    # Plots and visualizations (optional)
└── README.md


---

## 🚀 Getting Started

### 🔧 Requirements

* Python 3.8+
* pandas
* scikit-learn
* matplotlib
* numpy

Install dependencies:

```bash
pip install -r requirements.txt
```

### ▶️ Run the Model

1. Open the `wec-mlp-model.py` script.
2. Make sure all `.csv` files are in the same directory.
3. Run all cells or execute the script to train, evaluate, and visualize.

---

## 📌 Limitations & Future Work

* Current model is trained only on Perth data. Accuracy in other locations (like Sydney or PH) may be limited.
* Future work: integrate wave-specific parameters (e.g., direction, height) or retrain with PH-specific datasets.
* A Keras version could allow deeper architectures and GPU training.

---

## 💡 Acknowledgments

* Simulation data source: archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm
* Developed as a capstone project for DS100L: Applied Data Science.
* Model powered by scikit-learn.

---

## 📬 Contact


```

---

Would you like me to customize it further with your name or repo name? Or generate a `requirements.txt` too?
```
````
