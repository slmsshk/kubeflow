from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import joblib

def preprocess_data():
    iris = load_iris()
    X = iris.data
    y = iris.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    joblib.dump((X_train, X_test, y_train, y_test), '/tmp/iris_data.pkl')
    print("Data preprocessing done. Saved to /tmp/iris_data.pkl")

if __name__ == "__main__":
    preprocess_data()
