# Landslide
IoT based landslide monitoring and early prediction using LoRaWAN network. Rainfall, Soil Moisture, Geophone, Temperature and Humidity sensor integrated with ESP32 microcontroller to collect Real-time data and transfer to cloud via Raspberry Pi Gateway. Supervised, Unsupervised and Physic Informed neural network (PINN) models apply to predict lands
Signal Processing Pipeline
Raw sensor data is inherently noisy. This project implements advanced filtering techniques to ensure high-quality data before it reaches the predictive models:

Kalman Filters for real-time state estimation.

Butterworth Filters for frequency-based noise removal.

Wiener Filters for optimal noise reduction based on statistical approaches.

Machine Learning & Predictive Modeling
The forecasting engine evaluates the processed data using a combination of traditional and advanced AI algorithms implemented in Python and R:

Random Forest (RF) & Support Vector Machines (SVM): For robust classification and prediction based on historical and real-time sensor inputs.

Physics-Informed Neural Networks (PINN): To integrate the physical laws of soil mechanics directly into the deep learning architecture for highly accurate landslide predictions.
