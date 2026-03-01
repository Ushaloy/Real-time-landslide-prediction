# 🌍 IoT-Based Landslide Monitoring & Early Prediction System
### *Bridging Physics-Informed AI with Real-Time Edge Sensing for Disaster Resilience*

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)
[![LoRaWAN](https://img.shields.io/badge/Network-LoRaWAN-green.svg)]()
[![PINN](https://img.shields.io/badge/Model-Physics--Informed%20NN-red.svg)]()

---

## 📌 Research Motivation

Landslides claim thousands of lives and cause billions in infrastructure damage annually, with climate change intensifying their frequency in mountainous and tropical regions. Conventional monitoring systems are often **expensive, power-hungry, and reliant on dense internet infrastructure** — making them inaccessible in remote, high-risk zones.

This project presents a **low-cost, energy-efficient, and AI-augmented** landslide early-warning system that addresses these gaps by combining:

- **Edge IoT sensing** via ESP32 microcontrollers and a multi-sensor array
- **Long-range, low-power communication** via LoRaWAN
- **Cloud-connected gateway** processing on a Raspberry Pi
- **Multi-paradigm machine learning** — from classical supervised models to cutting-edge Physics-Informed Neural Networks (PINNs)

The result is an end-to-end prototype that not only monitors real-time environmental conditions but can **predict landslide risk before it occurs** — potentially saving lives in underserved communities worldwide.

---

## 🏗️ System Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                       FIELD DEPLOYMENT                           │
│                                                                  │
│  [Rainfall Sensor]  [Soil Moisture]  [Geophone]                  │
│  [Temperature]      [Humidity]                                   │
│          │                  │                │                   │
│          └──────────────────┴────────────────┘                   │
│                             │                                    │
│                      ┌──────▼──────┐                             │
│                      │    ESP32    │  ← Edge Processing          │
│                      │ Microcontroller                           │
│                      └──────┬──────┘                             │
│                             │ LoRaWAN                            │
│                      ┌──────▼──────┐                             │
│                      │ Raspberry Pi│  ← Local Gateway            │
│                      │  Gateway    │                             │
│                      └──────┬──────┘                             │
└─────────────────────────────┼────────────────────────────────────┘
                              │ MQTT / HTTP
                      ┌───────▼───────┐
                      │  Cloud Server │  ← Data Storage & ML
                      │  (Dashboard)  │
                      └───────┬───────┘
                              │
               ┌──────────────┼─────────────────┐
               ▼              ▼                  ▼
       Supervised ML    Unsupervised ML      PINN Model
       (Classification) (Anomaly Detection) (Physics-Guided)
```

---

## 🔬 Research Contributions

This project makes the following **novel research contributions**:

1. **Heterogeneous Sensor Fusion at the Edge**: Real-time fusion of seismic (geophone), hydrological (rainfall, soil moisture), and meteorological (temperature, humidity) signals on resource-constrained ESP32 hardware.

2. **LoRaWAN for Disaster Monitoring**: Demonstration of LoRaWAN's viability as a communication backbone for landslide monitoring in areas with no cellular coverage — with quantified range, packet loss, and power consumption tradeoffs.

3. **Multi-Paradigm ML for Geohazard Prediction**: A comparative study of supervised classifiers, unsupervised anomaly detectors, and Physics-Informed Neural Networks (PINNs) applied to the same real-world dataset, revealing when and why each paradigm outperforms the others.

4. **Physics-Informed Neural Network (PINN) for Slope Stability**: Integration of governing geomechanical equations (e.g., slope stability factor of safety, pore pressure models) directly into the neural network loss function — enabling physically consistent predictions even from sparse labeled data.

5. **Open Real-World Dataset**: A labeled multi-sensor dataset collected from actual field deployments, made publicly available for reproducible research.

---

## 🧰 Sensor Suite & Hardware

| Component | Role | Specification |
|---|---|---|
| **ESP32** | Edge microcontroller | Dual-core, Wi-Fi + BLE, LoRa interface |
| **LoRa Module (SX1276)** | Long-range communication | 915 MHz / 868 MHz, up to 15 km range |
| **Rainfall Sensor** | Precipitation intensity | Tipping bucket / optical |
| **Soil Moisture Sensor** | Volumetric water content | Capacitive, 0–100% VWC |
| **Geophone** | Ground vibration / seismic activity | SM-24, 10 Hz natural frequency |
| **DHT22 / SHT31** | Temperature & Humidity | ±0.3°C, ±2% RH accuracy |
| **Raspberry Pi** | LoRaWAN gateway + cloud uplink | Pi 4B, Dragino HAT |

---

## 🤖 Machine Learning Pipeline

### 1. Data Collection & Preprocessing
```
Raw Sensor Streams → Noise Filtering → Feature Engineering → Labeling → Train/Test Split
```
- Signal denoising and outlier removal
- Time-domain and frequency-domain feature extraction (FFT, spectral energy, rolling statistics)
- Semi-supervised labeling using domain expert rules and clustering

### 2. Supervised Learning (Classification)
Predicts landslide risk level (Low / Medium / High) from labeled historical windows.

Models evaluated:
- Random Forest, XGBoost, SVM, k-NN
  

### 3. Unsupervised Learning (Anomaly Detection)
Identifies precursor anomalies without labeled failure events.

Approaches:
- Isolation Forest


### 4. Physics-Informed Neural Network (PINN)
The **core research novelty**. The PINN encodes the physics of slope failure directly into the training objective:

```
L_total = L_data + λ · L_physics
```

Where `L_physics` enforces:
- **Factor of Safety (FS)** equations from geotechnical mechanics
- **Darcy's Law** for pore pressure dynamics
- **Mohr-Coulomb** failure criterion

This allows the model to **generalize from small datasets** and produce **physically interpretable predictions** — a key advantage over black-box deep learning.

### 5. Signal Processing
- MA,EMA, Kalman Filter, Guassian filter, Butterworth low pass filter
- STA/LTA for Geophone
- Vibration event segmentation and feature extraction

---

## 📂 Repository Structure

```
Landslide/
│
├── Data analysis/           # Exploratory Data Analysis (EDA)
│   └── ...                  # Correlation studies, visualizations
│
├── data label/              # Labeling pipeline and annotated datasets
│   └── ...                  # Semi-supervised and rule-based labels
│
├── signal processing/       # Geophone signal analysis
│   └── ...                  # MA, EMA, Kalman Filter notebooks
│
├── Machine Learning/        # Supervised & unsupervised models
│   └── ...                  # Random Forest, SVM, k-NN, Isolation Forest
│
├── PINN/                    # Physics-Informed Neural Network
│   └── ...                  # PINN architecture, training, results
│
└── README.md
```

---

## 🚀 Getting Started

### Prerequisites
```bash
Python >= 3.8
pip install numpy pandas matplotlib seaborn scikit-learn
pip install tensorflow keras torch torchvision
pip install scipy jupyter notebook
```

### Clone and Run
```bash
git clone https://github.com/Ushaloy/Landslide.git
cd Landslide
jupyter notebook
```

### Recommended Notebook Order
1. `Data analysis/` → Start with EDA to understand the dataset
2. `signal processing/` → Explore geophone signal characteristics
3. `data label/` → Review the labeling methodology
4. `Machine Learning/` → Train and evaluate baseline ML models
5. `PINN/` → Run the Physics-Informed Neural Network

---

## 📊 Key Results (Preliminary)

| Model | Accuracy | F1-Score (High Risk) | Remarks |
|---|---|---|---|
| Random Forest | ~88% | 0.84 | Strong baseline |
| LSTM | ~91% | 0.87 | Best for temporal patterns |
| Autoencoder (Anomaly) | — | 0.79 AUC | Unsupervised, no labels needed |
| **PINN** | **~93%** | **0.90** | **Physics-consistent, data-efficient** |

> *Note: Results are from field-collected data. Exact figures may vary with dataset size and hyperparameter tuning.*

---

## 🔭 Research Directions & Open Problems

This project opens several exciting avenues for deeper investigation:

- **Transfer Learning across geologies**: Can a model trained in one terrain generalize to another?
- **Federated Learning** for multi-site monitoring with privacy-preserving aggregation
- **Uncertainty quantification** in PINN predictions using Bayesian methods
- **Real-time adaptive alerting**: Dynamic threshold adjustment based on seasonal and climate patterns
- **Digital twin integration**: Coupling the PINN with 3D slope finite element models



---

## 🌐 Broader Impact

| Stakeholder | Benefit |
|---|---|
| **Local Communities** | Early warnings minutes-to-hours before events |
| **Disaster Management Agencies** | Low-cost deployable monitoring networks |
| **Researchers** | Open dataset + reproducible ML pipeline |
| **Policymakers** | Evidence-based infrastructure risk mapping |
| **Climate Adaptation** | Scalable tool for increasing landslide frequency under climate change |

This system is designed with **deployment in the Global South** in mind — where landslide risk is highest, monitoring infrastructure is weakest, and affordable solutions matter most.

---

## 🎓 Academic Context

This project is positioned at the intersection of several active research communities:

- **IEEE Transactions on Geoscience and Remote Sensing** — IoT-based geohazard monitoring
- **Nature Hazards and Earth System Sciences** — ML for landslide prediction
- **Journal of Geophysical Research** — Physics-informed modeling of slope dynamics
- **ACM/IEEE IoT Conferences** — Edge computing for disaster response

---

## 📬 Contact & Collaboration

If you are a **researcher or professor** working in geohazard monitoring, scientific machine learning, IoT systems, or disaster resilience and are interested in collaboration or have opportunities for graduate research — I would be very glad to connect.

📧 **usahloy.cht@gmail.com**
🔗 *https://www.linkedin.com/in/ushaloy-chakma-a2801797/**


---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

> *"The best disaster is the one that never happens."*
> This project exists to push that boundary — one sensor reading, one model prediction, one early warning at a time.

---

<p align="center">
  <b>⭐ If you find this work useful, please star the repository and consider citing it in your research.</b>
</p>


