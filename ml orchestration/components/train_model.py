from sklearn.ensemble import RandomForestClassifier
import joblib

def train_model():
    X_train, X_test, y_train, y_test = joblib.load('/tmp/iris_data.pkl')

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    joblib.dump(model, '/tmp/iris_model.pkl')
    print("Model training completed. Saved to /tmp/iris_model.pkl")

if __name__ == "__main__":
    train_model()
