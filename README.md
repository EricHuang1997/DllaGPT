# DllaGPT

DllaGPT is a project that uses advanced machine learning techniques for dynamic lifelong learning-based aspect-based sentiment analysis (ABSA) across multiple domains. This project includes a directory of datasets and several scripts for training, testing, and evaluating the model. To run the main operations, please use the `tentative.py` file as the entry point.

---

## Directory Structure

### 1. **Data_Directory**  
This directory contains the training, validation, and testing datasets for multiple domains (e.g., finance, laptop, restaurant, tweets). The datasets are organized as follows:

- `finance-train.json` / `finance-val.json` / `finance-test.json`  
- `laptop-train.json` / `laptop-val.json` / `laptop-test.json`  
- `restaurant-train.json` / `restaurant-val.json` / `restaurant-test.json`  
- `tweets-train.json` / `tweets-val.json` / `tweets-test.json`  
- Multi-domain datasets are also provided in the format `*-multi.json` for domain-specific experiments.

---

### 2. **Model_Directory**  
Contains the model training log and final saved model. Files include:
- `log_train.txt` – Training logs.
- `model-finish` – Final model saved after training.

---

### 3. **Scripts and Configuration Files**  

#### **Key Python Scripts**  
- `tentative.py` – **Main script for running the model**.  
- `tentative.py` – Script for training the model aslo.  
- `test.py` – Script for testing the trained model.
- `preprocess.py` – Data preprocessing utilities.  
- `settings.py` – Configuration settings for the project.  
- `metrics.py` – Script for calculating evaluation metrics.  
- `loss_scaler.py`, `regularizers.py`, `parallel.py` – Various utilities for improving performance and regularization.

#### **Shell Scripts**  
- `Tentative.sh` – A shell script for batch running the `tentative.py` file.

#### **Supporting Files**  
- `requirements.txt` – Python dependencies required for running the project.  
- `README.md` – Project documentation (this file).

---

## Installation

1. Clone the repository to your local machine.
2. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
3. Activate the virtual environment:
   - On Windows: `.\venv\Scripts\activate`
   - On macOS/Linux: `source venv/bin/activate`
4. Install required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage Instructions

### Running the Model
To train and test the model, use the following commands:

1. **Run the main entry point:**  
   ```bash
   python tentative.py
   ```

2. **Training:**  
   If you want to train the model manually, run:  
   ```bash
   python Tentative.py
   ```

3. **Testing:**  
   Run the test script:  
   ```bash
   python test_.py
   ```

---

## Data Files
The datasets provided are in JSON format, with each file corresponding to different domains or tasks. Ensure that all dataset files are correctly formatted before running the scripts.

---

## Contact
For any issues or inquiries related to this project, please reach out to the project maintainers(erichuangemail@163.com).

---
