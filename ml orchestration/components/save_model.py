import joblib
import os
from datetime import datetime

def save_model():
    model = joblib.load('/tmp/iris_model.pkl')
    os.makedirs('/mnt/models', exist_ok=True)

    version = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = f'/mnt/models/iris_model_{version}.pkl'
    joblib.dump(model, path)
    print(f"Model saved: {path}")

if __name__ == "__main__":
    save_model()
