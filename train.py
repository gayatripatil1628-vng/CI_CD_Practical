
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score

def train_model():
    iris = load_iris()

    X_train, X_test, y_train, y_test = train_test_split(
        iris.data,
        iris.target,
        test_size=0.2,
        random_state=42
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions)

    print("Model Training Successful!")
    print("Accuracy:", accuracy)

    with open("build_report.txt", "w") as file:
        file.write("CI Build Report\n")
        file.write("Model Training Successful\n")
        file.write(f"Accuracy: {accuracy}\n")
        file.write("Status: PASSED\n")

    return accuracy

if __name__ == "__main__":
    train_model()
