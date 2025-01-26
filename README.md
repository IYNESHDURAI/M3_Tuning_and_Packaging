# M3: Model Experimentation and Packaging

## Objective
This project focuses on training a machine learning model, performing hyperparameter tuning using Optuna, and deploying the best-performing model using Flask and Docker.

## Project Structure

```plaintext
m3_project/
├── model_tuning.py       # Hyperparameter tuning and model training
├── app.py                # Flask API for model prediction
├── Dockerfile            # Docker configuration for the app
├── requirements.txt      # Project dependencies
├── best_model.pkl        # Trained model
└── README.md             # Project documentation
```

## Steps

### 1. Set Up Python Environment
Create and activate a virtual environment:
```bash
python -m venv env
# Activate:
# Windows: env\Scripts\activate
# Mac/Linux: source env/bin/activate
```

Install dependencies:
```bash
pip install -r requirements.txt
```

### 2. Hyperparameter Tuning
Run `model_tuning.py` to tune hyperparameters using Optuna and train the best model:
```bash
python model_tuning.py
```

### 3. Flask Model Deployment
The trained model is served using Flask in `app.py`. Run the Flask app:
```bash
python app.py
```

### 4. Dockerize the Application
Build and run the Docker container:
```bash
docker build -t model-flask-app .
docker run -d -p 5000:5000 model-flask-app
```

### 5. Test the API
Use `Postman` or `curl` to test the API:
```bash
curl -X POST -H "Content-Type: application/json" \
-d '{"features": [13.2, 1.78, 2.14, 11.2, 100.0, 2.65, 2.76, 0.26, 1.28, 4.38, 1.05, 3.4, 1050]}' \
http://127.0.0.1:5000/predict
```

## Deliverables
1. **Report**: Hyperparameter tuning process, best parameters, and model performance.
2. **Code**:
   - `model_tuning.py`
   - `app.py`
   - `Dockerfile`
3. **Screenshots**: Docker container running and prediction requests/responses.

## Requirements
- **Python 3.9+**
- Libraries: `flask`, `scikit-learn`, `optuna`, `pandas`, `numpy`, `matplotlib`, `gunicorn`

Install dependencies:
```bash
pip install -r requirements.txt
```

## License
MIT License
```

This `README.md` provides a concise overview and instructions for setting up, running, and testing the project. Let me know if you need further adjustments!
