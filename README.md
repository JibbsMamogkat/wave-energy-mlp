
# 🌊 Wave Farm Energy Prediction with Neural Networks

This project applies machine learning to predict the total power output of wave energy converter (WEC) farms based solely on buoy layout configuration. Using simulation-based datasets from Perth and Sydney, a neural network was trained to model the relationship between layout and energy generation — with future application in the Philippine context.

---

## 📌 Overview

- **Goal**: Predict total energy output of a WEC layout using machine learning
- **Dataset**: [UCI Large-Scale Wave Energy Farm](https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm)
- **Model**: `MLPRegressor` (Neural Network)
- **Input Features**: X1–Y49 (98 total for 49-buoy layouts)
- **Target**: `Total_Power`
- **Main Dataset**: `WEC_Perth_49.csv` (36,043 layouts)
- **SDG Focus**: UN SDG 7 - Affordable and Clean Energy

---

## ⚙️ Features

- ✅ Trains a neural network using layout-only inputs
- 📊 Evaluates accuracy using MSE, MAE, and Percent Error
- 🌍 Generalizes to Sydney layouts and a simulated Philippine layout
- 🔁 Compares ReLU, tanh, and logistic activation functions
- 📈 Visualizes model performance

---
## 📊 Model Performance Summary

| **Model**           | **Training Dataset**     | **Test Dataset**        | **MAE (W)**   | **Percent Error** | **Notes**                            |
|---------------------|--------------------------|--------------------------|---------------|--------------------|----------------------------------------|
| 49-Buoy Model       | Perth (36,043 samples)   | Perth                    | 25,062        | 0.63% ✅           | Best overall performance               |
| 49-Buoy Model       | Perth                    | Sydney (49-buoy)         | 166,534       | 4.14%              | Moderate generalization drop           |
| 49-Buoy Model       | Perth                    | Sample PH Layout         | —             | —                  | Predicted: **5,350,615.65 W**          |
| 100-Buoy Model      | Perth (7,277 samples)    | Perth                    | 260,840       | 3.67%              | Weaker due to low data volume          |
| 100-Buoy Model      | Perth                    | Sydney (100-buoy)        | 729,819       | 10.18% ❌          | Poor generalization and prediction bias|


---

## 📁 Project Structure

```bash
FINAL-PROJECT/
├── wec-mlp-model.py               # Main Python script
├── sample_philippine_layout.csv  # 7x7 grid layout (60m spacing)
├── wave-farms-study.pdf          # Reference research paper
├── WEC.zip                       # Compressed version of datasets
├── large-scale+wave+energy+farm.zip  # Raw reference zip (if needed)
├── WEC_Perth_49.csv              # Main training dataset
├── WEC_Sydney_49.csv             # Generalization test
├── WEC_Perth_100.csv             # Optional advanced dataset
├── WEC_Sydney_100.csv            # Optional test set for 100-buoy
````

---

## ▶️ How to Run

### 1. Install dependencies:

```bash
pip install pandas scikit-learn matplotlib numpy
```

### 2. Open and run `wec-mlp-model.py`

Make sure all `.csv` files are in the same directory.

---

## 🔬 Model Architecture

* Model: `MLPRegressor`
* Hidden Layers: (128, 64)
* Activation: ReLU (best-performing)
* Learning Rate: 0.0005
* Iterations: 1000
* Scaler: `StandardScaler`

---

## 🧪 Experiments Conducted

* ✅ Hyperparameter tuning (learning rate, iterations)
* ✅ Activation function comparison (`relu`, `tanh`, `logistic`)
* ✅ Generalization test on Sydney 49-buoy data
* ✅ Sample layout simulation for Philippine use case
* ✅ Predicted vs Actual plots for all model runs

---

## 📌 Limitations & Future Work

* Model trained only on Perth layouts; generalization is limited
* Wave-specific inputs (e.g., direction, height) not included
* Future work: train model using PH-specific simulated wave data
* Explore deeper models using Keras or PyTorch

---

## 💡 Acknowledgments

* Dataset Source: [UCI Machine Learning Repository – Large-Scale Wave Energy Farm](https://archive.ics.uci.edu/dataset/882/large-scale+wave+energy+farm)
* Original study provided simulation data for both Perth and Sydney wave regimes
* Project created for **DS100L: Applied Data Science**
* Powered by `scikit-learn`, `pandas`, and `matplotlib`

---

## 📬 Contact

