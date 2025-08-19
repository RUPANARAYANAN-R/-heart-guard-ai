# ❤️‍🩹 HeartGuard AI

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Build Status](https://img.shields.io/badge/build-passing-brightgreen.svg)](https://github.com/)
[![Python Version](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Leveraging **machine learning** to predict the risk of **cardiovascular disease**, providing an early warning system for proactive healthcare.

---

## Table of Contents

- [About The Project](#about-the-project)
- [Key Features](#key-features)
- [How It Works](#how-it-works)
- [Technology Stack](#technology-stack)
- [Getting Started](#getting-started)
- [Project Usage](#project-usage)
- [Model Training](#model-training)
- [API Endpoints](#api-endpoints)
- [Project Roadmap](#project-roadmap)
- [Project Contributing](#project-contributing)
- [Project License](#project-license)
- [Contact Information](#contact-information)
- [Important Disclaimer](#important-disclaimer)

---

## About The Project

**Cardiovascular diseases (CVDs)** are the leading cause of death globally. Early detection and risk stratification are crucial for effective prevention and management. **HeartGuard AI** is a project designed to address this challenge by using a machine learning model to predict a patient's risk of heart disease based on a set of medical and demographic attributes.

This tool provides a simple **REST API** and an interactive **dashboard** for users to input patient data and receive an instant risk assessment. The core of this project is a trained classification model that has learned patterns from historical patient data.

**Dataset:** This model was trained on the **[UCI Heart Disease Dataset](https://archive.ics.uci.edu/ml/datasets/heart+disease)**, a well-known benchmark in the machine learning community.

---

## Key Features

✨ **High-Accuracy Prediction:** Utilizes a **Gradient Boosting Classifier** for robust and accurate risk prediction.

✨ **RESTful API:** A clean, fast, and scalable API built with **FastAPI** for easy integration into other applications.

✨ **Interactive Dashboard:** A user-friendly web interface created with **Streamlit** for real-time predictions and data visualization.

✨ **Feature Importance:** Provides insights into which factors are most influential in predicting heart disease risk.

✨ **Containerized:** **Docker** support for easy setup, deployment, and scalability.

---

## How It Works

The architecture separates the **offline training process** from the **online prediction service**.

### Offline Training Pipeline

    [Raw Dataset] --> [1. Data Cleaning & Preprocessing] --> [2. Model Training] --> [Saved Model File]


### Online Prediction Pipeline
 
    [User Input] --> [FastAPI Backend] --> [Load Saved Model] --> [Make Prediction] --> [Display Result]


---

## Technology Stack

This project is built with **modern, open-source technologies**.

- **Backend:** Python, FastAPI
- **Machine Learning:** Scikit-learn, Pandas, NumPy
- **Dashboard:** Streamlit
- **Deployment:** Docker, Uvicorn
- **Code Quality:** Black, isort

---

## Getting Started

Follow these instructions to get a local copy up and running.

### Prerequisites

- Python 3.9+ and `pip`
- Git
- Docker (optional, for containerized deployment)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/heartguard-ai.git](https://github.com/your-username/heartguard-ai.git)
    cd heartguard-ai
    ```

2.  **Create and activate a virtual environment:**
    ```bash
    # For macOS/Linux
    python3 -m venv venv
    source venv/bin/activate
    
    # For Windows
    python -m venv venv
    .\venv\Scripts\activate
    ```

3.  **Install the required dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

4.  **Set up environment variables:**
    Create a `.env` file in the root directory by copying the example file.
    ```bash
    cp .env.example .env
    ```
    (No changes are needed in `.env` for the default setup.)

---

## Project Usage

### Running the API

To start the **FastAPI server**, run the following command from the root directory:
```bash
uvicorn app.main:app --reload
The API will be available at http://127.0.0.1:8000. You can access the interactive API documentation (Swagger UI) at http://127.0.0.1:8000/docs.

## Running the Dashboard
To launch the Streamlit dashboard, run:

## Bash

streamlit run dashboard.py
The dashboard will open in your web browser, usually at http://localhost:8501.

## Using Docker (Optional)
Build and run the application using Docker Compose for a containerized setup.

## Bash

docker-compose up --build
This will start both the FastAPI backend and the Streamlit frontend.
```
## Model Training
To train the model from scratch, you can run the training script. This script will process the data, train the model, evaluate it, and save the final model artifact.

    Bash

    python scripts/train_model.py
API Endpoints
The primary endpoint for prediction is: POST /predict.

Description: Accepts patient data and returns a heart disease risk prediction.

    Body (JSON):

    JSON

    {
    "age": 52, "sex": 1, "cp": 0, "trestbps": 125, "chol": 212, "fbs": 0,
    "restecg": 1, "thalach": 168, "exang": 0, "oldpeak": 1.0, "slope": 2,
    "ca": 2, "thal": 3
    }
    Success Response (200):

    JSON

    {
    "prediction": 0,
    "prediction_label": "No Disease",
    "probability": 0.92
    }
(Note: prediction: 0 means low risk and prediction: 1 means high risk.)

---

## Project Roadmap
[ ] Phase 1: Core Functionality

    [x] Model Training and Evaluation

    [x] REST API with FastAPI

    [x] Interactive Dashboard with Streamlit

    [x] Dockerization

[ ] Phase 2: Enhancements

    [ ] Add user authentication.

    [ ] Integrate with a database to store prediction history.

    [ ] Perform hyperparameter tuning.

    [ ] Expand dataset.

    [ ] Phase 3: Advanced Features

    [ ] Incorporate explainable AI (XAI) techniques (e.g., SHAP, LIME).

    [ ] Explore deep learning models for ECG signal analysis.

    [ ] Deploy to a cloud platform (AWS, GCP, Azure).

See the open issues for a full list of proposed features and known issues.

---

## Project Contributing
Contributions are greatly appreciated. Please follow these steps:

Fork the Project

    Create your Feature Branch (git checkout -b feature/AmazingFeature)

    Commit your Changes (git commit -m 'Add some AmazingFeature')

    Push to the Branch (git push origin feature/AmazingFeature)

Open a Pull Request

---

## Project License
Distributed under the MIT License. See LICENSE.txt for more information.

---

## Contact Information
### Rupa Narayanan

### Email: rupanarayanan333@gmail.com

---

### Project Link: https://github.com/your-username/heartguard-ai

---

## Important Disclaimer

### This project is for educational and research purposes only. The predictions made by HeartGuard AI are not a substitute for professional medical advice, diagnosis, or treatment. Always seek the advice of a qualified health provider with any questions you may have regarding a medical condition.
