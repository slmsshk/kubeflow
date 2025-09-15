from sklearn.metrics import accuracy_score
import joblib

def evaluate_model():
    X_train, X_test, y_train, y_test = joblib.load('/tmp/iris_data.pkl')
    model = joblib.load('/tmp/iris_model.pkl')

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save metric for Kubeflow to use in Conditions
    with open('/tmp/accuracy.txt', 'w') as f:
        f.write(str(acc))

    print(f"Model Accuracy: {acc}")

if __name__ == "__main__":
    evaluate_model()
