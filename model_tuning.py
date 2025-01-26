import optuna
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier

# Load the dataset
data = load_wine()
X, y = data.data, data.target

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the objective function for Optuna
def objective(trial):
    # Hyperparameters to tune
    n_estimators = trial.suggest_int("n_estimators", 10, 200)
    max_depth = trial.suggest_int("max_depth", 2, 32, log=True)
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)

    # Model with tuned hyperparameters
    clf = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )
    # Cross-validation score
    score = cross_val_score(clf, X_train, y_train, cv=3, scoring="accuracy").mean()
    return score

# Perform hyperparameter tuning
study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=50)

# Print best parameters
print("Best Parameters:", study.best_params)

# Train final model with best parameters
best_params = study.best_params
best_model = RandomForestClassifier(**best_params, random_state=42)
best_model.fit(X_train, y_train)

# Save the model using joblib
import joblib
joblib.dump(best_model, "best_model.pkl")
print("Model saved as best_model.pkl")
