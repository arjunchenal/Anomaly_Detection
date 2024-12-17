# Anomaly_Detection

# LSTM Time Series Anomaly Detection

This repository provides an LSTM-based model for anomaly detection in time series data. The model is designed to process sequential trace data, identify anomalies, and evaluate its performance using precision, recall, and F1 metrics.

---

## Features

- **Data Loading & Preprocessing**: The model reads and preprocesses trace data, normalizes it using MinMaxScaler, and structures it for LSTM input.
- **Train-Validation Split**: The dataset is split into training and validation sets for model evaluation.
- **LSTM Model**: Two-layer LSTM model with dropout for regularization and dense output for predictions.
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

Install the required packages using the below command:

```bash
pip install tensorflow numpy pandas scikit-learn matplotlib
```

---


## File Structure

- **LSTM_Model.ipynb**: Jupyter Notebook file containing the full implementation of the LSTM anomaly detection model.

---

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
     jupyter notebook LSTM_Model.ipynb
     ```
   - Execute the cells sequentially.

4. **Train the Model**:
   - The model will train on the data and provide training and validation loss curves.
   - Adjust the number of epochs based on the plotted curves to avoid overfitting.

5. **Evaluate the Model**:
   - Precision, Recall, F1 Score, and Confusion Matrix will be printed.
   - Visualizations of predictions and anomalies will be displayed.

---

## Output

- Training and validation loss curves.
- Anomaly detection results with identified anomalies in trace data.
- Evaluation metrics (Precision, Recall, F1 Score).
- Confusion matrix visualization.

---

## Future Improvements

- Include Early Stopping to optimize training.
- Use additional anomaly detection thresholds dynamically.
- Add support for multi-threaded trace data.

---

## License

This project is licensed under the MIT License.

---
