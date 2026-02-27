# Landslide Monitoring and Early Prediction 
IoT based landslide monitoring and early prediction using LoRaWAN network. Rainfall, Soil Moisture, Geophone, Temperature and Humidity sensor integrated with ESP32 microcontroller to collect Real-time data and transfer to cloud via Raspberry Pi Gateway. Supervised, Unsupervised and Physic Informed neural network (PINN) models apply to predict lands
🛠️ Modules & Methodologies
1. Data & Analysis (/data, /data_analysis, /labeling)
Data Handling: Ingestion of raw time-series data from the IoT gateway.

Analysis: Statistical evaluation of soil parameters and precipitation using Python (Pandas/NumPy) and R (ggplot2).

Labeling: Automated and manual annotation workflows to classify stable vs. unstable ground conditions based on historical data.

2. Signal Processing (/signal_processing)
Raw field data is filtered to remove environmental noise and sensor artifacts before modeling:

EMA (Exponential Moving Average): For smoothing short-term fluctuations in moisture and rain gauges.

Kalman Filter: For real-time state estimation and predicting the true state of dynamic soil sensors.

Butterworth Filter: For frequency-domain filtering to isolate meaningful geological and environmental shifts.

3. Machine Learning & Anomaly Detection (/models_ml)
A suite of algorithms to classify conditions and detect outliers:

Random Forest (RF): Ensemble learning for robust classification of multi-sensor data.

Support Vector Machine (SVM): High-dimensional boundary mapping to separate safe vs. hazardous states.

k-Nearest Neighbors (k-NN): Instance-based learning for localized pattern recognition.

Isolation Forest: Unsupervised anomaly detection, highly effective for identifying sudden, abnormal sensor spikes that precede slope failure.

4. Physics-Informed Neural Networks (/models_pinn)
Deep learning models constrained by the physical laws of soil mechanics and hydrology. Instead of relying purely on data, the PINN incorporates differential equations governing soil sheer strength and water infiltration, ensuring predictions remain physically viable even with sparse sensor data.
