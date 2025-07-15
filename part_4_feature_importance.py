# part_4_feature_importance.py

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

data = pd.read_csv("synthetic_groundwater_data.csv")

land_use_encoder = LabelEncoder()
data["land_use_encoded"] = land_use_encoder.fit_transform(data["land_use"])

target_encoder = LabelEncoder()
data["recharge_encoded"] = target_encoder.fit_transform(data["recharge_potential"])

features = ["slope", "elevation", "soil_permeability", "rainfall", "drainage_density", "depth_to_water_table", "land_use_encoded"]

X = data[features]
y = data["recharge_encoded"]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

rf = RandomForestClassifier(random_state = 42)
rf.fit(X_scaled, y)

importances = rf.feature_importances_
feature_importance_df = pd.DataFrame({"Feature": features, "Importance": importances})
feature_importance_df.sort_values(by = "Importance", ascending = False, inplace = True)

plt.figure(figsize = (8, 6))
sns.barplot(x = "Importance", y = "Feature", data = feature_importance_df)
plt.title("Feature Importance - Random Forest")
plt.tight_layout()
plt.show()

# Prediction function
def predict_recharge(slope, elevation, soil_permeability, rainfall, drainage_density, depth_to_water_table, land_use_category):
    
    # Encode land use category
    land_use_encoded = land_use_encoder.transform([land_use_category])[0]
    
    # Prepare input as DataFrame with correct feature names
    input_data = pd.DataFrame([[slope, elevation, soil_permeability, rainfall, drainage_density, depth_to_water_table, land_use_encoded]], columns = features)
    
    # Scale input
    input_scaled = scaler.transform(input_data)
    
    # Predict class and decode label
    class_id = rf.predict(input_scaled)[0]
    return target_encoder.inverse_transform([class_id])[0]

# Example usage
result = predict_recharge(5, 300, 3, 900, 0.5, 20, "agriculture")
print("Predicted Recharge Potential:", result)
