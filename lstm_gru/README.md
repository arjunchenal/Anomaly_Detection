# Anomaly_Detection using LSTM and GRU Models

This repository provides comprehensive implementation for anomaly detection in the time series data using LSTM and GRU models. It maily focus only two primary scenarios.
1. ID model
2. ID and Timestamp as Tuple Model
Significant accuracy variations were observed, a drastic accuracy drop in ID and Timestamp as tuple model compared to ID model.

---

## Features

- **Data Loading & Preprocessing**: Handling trace data with normalization using MinMaxScaler. It is suitable for both LSTM and GRU.
- **Train-Validation Split**: The dataset is split into training and validation sets for model evaluation.
- **Models Implemented**: Architecture configurations tested with layers of 128,64 and 32 units including dropout for regularization.
- **Anomaly Detection**: Threshold-based anomaly detection is implemented using reconstruction error.
- **Evaluation Metrics**: Precision, Recall, F1 Score, and Confusion Matrix visualization are included.
- **Training & Validation Curves**: Loss curves are plotted to monitor training performance and detect overfitting.

---


## Prerequisites

Make sure the following libraries are installed in your environment:

- Python 3.x
- TensorFlow 2.x
- NumPy
- Pandas
- Scikit-learn
- Matplotlib

Install the required packages using below command:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---

## Repository Structure
- **LSTM_ID_MODEL**: Jupyter Notebook file containing the full implementation of the LSTM anomaly detection model.
- **LSTM_ID_TIMESTAMP_Model**: For training and evaluation using ID Timestamp as Tuple model.
- **GRU_ID_MODEL**: Jupyter Notebook file containing the full implementation of the GRU anomaly detection model.
- **GRU_ID_TIMESTAMP_Model**: For training and evaluation using ID Timestamp as Tuple model.
- **lstm_id_test**: Jupyter Notebook file for testing phase. It loads the trained model(LSTM) and scaler for testing phase.
- **lstm_id_timestamp_test**: Jupyter Notebook file for testing phase. It loads the trained model(LSTM) and scaler for testing phase.
- **gru_id_test**: Jupyter Notebook file for testing phase. It loads the trained model(GRU) and scaler for testing phase.
- **gru_id_timestamp_test**: Jupyter Notebook file for testing phase. It loads the trained model(GRU) and scaler for testing phase.
- **libraries folder**: This folder contains the anomaly_detection.py and utils.py file. Helper functions are included in these files.
- **anomaly_detection.py**:Contains the function which is used for test data detections, merging detected sequences and comparing with ground truth data.
- **utils.py**: Contains utility functions for read_trace, loading data.

## How to Run

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/arjuncee123/Anomaly_Detection.git
   cd Anomaly_Detection
   ```

2. **Prepare Data**:
   - Place your zipped trace data file (`theft_protection.zip`) in the project directory.
   - Extract the data and structure it as required by the notebook.

3. **Run the Notebook**:
   - Open the Jupyter Notebook:
     ```bash
     jupyter notebook LSTM_ID_MODEL.ipynb
     ```
   - Execute the cells sequentially.

4. **Train the Model**:
   - The model will train on the data and provide the accuracy of the model.

5. **Saving the Model and Scaler**:
   - Using trained model and scaler for evaluation for test data.   

5. **Evaluate the Model**:
   - Precision, Recall, F1 Score, and Confusion Matrix will be printed.
   - Visualizations of predictions and anomalies will be displayed.

---

## Key Observations
1. ID model gives higher accuracy compared to the ID timestamp tuple model.

---

## License

This project is licensed under the MIT License.

---
