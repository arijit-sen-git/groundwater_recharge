# part_2_preprocessing_eda.py

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("synthetic_groundwater_data.csv")

print(data.head())
print(data.info())

data["land_use_encoded"] = LabelEncoder().fit_transform(data["land_use"])

target_encoder = LabelEncoder()
data["recharge_encoded"] = target_encoder.fit_transform(data["recharge_potential"])

label_map = dict(zip(target_encoder.classes_, target_encoder.transform(target_encoder.classes_)))
print("Label Mapping:", label_map)

features = ["slope", "elevation", "soil_permeability", "rainfall", "drainage_density", "depth_to_water_table", "land_use_encoded"]

X = data[features]
y = data["recharge_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

sns.countplot(x = "recharge_potential", data = data)
plt.title("Recharge Potential Class Distribution")
plt.show()

correlation_matrix = data[features + ["recharge_encoded"]].corr()

plt.figure(figsize = (10, 6))
sns.heatmap(correlation_matrix, annot = True, cmap = "coolwarm")
plt.title("Correlation Heatmap")
plt.show()

for feature in ["slope", "elevation", "rainfall", "depth_to_water_table"]:
    plt.figure()
    sns.boxplot(x="recharge_potential", y = feature, data = data)
    plt.title("{} vs Recharge Potential".format(feature))
    plt.show()