# part_3_model_training.py

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

data = pd.read_csv("synthetic_groundwater_data.csv")

data["land_use_encoded"] = LabelEncoder().fit_transform(data["land_use"])
data["recharge_encoded"] = LabelEncoder().fit_transform(data["recharge_potential"])

features = ["slope", "elevation", "soil_permeability", "rainfall", "drainage_density", "depth_to_water_table", "land_use_encoded"]

X = data[features]
y = data["recharge_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size = 0.2, random_state = 42, stratify = y)

models = {
    "Random Forest": RandomForestClassifier(random_state = 42),
    "SVM": SVC(kernel = "rbf", C = 1),
    "KNN": KNeighborsClassifier(n_neighbors = 5),
    "Logistic Regression": LogisticRegression(max_iter = 1000)
}

results = {}

for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    results[name] = acc

    print("\nModel: {}".format(name))
    print("Accuracy: ", acc)
    print("Classification Report: \n", classification_report(y_test, y_pred))
    
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("{} - Confusion Matrix".format(name))
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

plt.figure(figsize=(8, 5))
sns.barplot(x = list(results.keys()), y = list(results.values()))
plt.title("Model Accuracy Comparison")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.show()