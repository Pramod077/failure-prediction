import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

# Load the data
df = pd.read_csv('CIRCA.csv')

# Display basic info
print("Dataset shape:", df.shape)
print("\nFirst few rows:")
print(df.head())

# Prepare features and targets
X = df[['load_lbf', 'thickness_mm']]
y_stress = df['max_stress_ksi']
y_deflection = df['max_deflection_inch']

# Split data into training and testing sets (80-20 split)
X_train, X_test, y_stress_train, y_stress_test = train_test_split(X, y_stress, test_size=0.2, random_state=42)
_, _, y_deflection_train, y_deflection_test = train_test_split(X, y_deflection, test_size=0.2, random_state=42)

# Train linear regression model for stress
stress_model = LinearRegression()
stress_model.fit(X_train, y_stress_train)

# Train linear regression model for deflection
deflection_model = LinearRegression()
deflection_model.fit(X_train, y_deflection_train)

# Make predictions on test set
y_stress_pred = stress_model.predict(X_test)
y_deflection_pred = deflection_model.predict(X_test)

# Evaluate models
print("\n" + "="*60)
print("STRESS MODEL PERFORMANCE")
print("="*60)
print(f"R² Score: {r2_score(y_stress_test, y_stress_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_stress_test, y_stress_pred)):.4f}")
print(f"Coefficients: load_lbf={stress_model.coef_[0]:.6f}, thickness_mm={stress_model.coef_[1]:.6f}")
print(f"Intercept: {stress_model.intercept_:.6f}")

print("\n" + "="*60)
print("DEFLECTION MODEL PERFORMANCE")
print("="*60)
print(f"R² Score: {r2_score(y_deflection_test, y_deflection_pred):.4f}")
print(f"RMSE: {np.sqrt(mean_squared_error(y_deflection_test, y_deflection_pred)):.4f}")
print(f"Coefficients: load_lbf={deflection_model.coef_[0]:.6f}, thickness_mm={deflection_model.coef_[1]:.6f}")
print(f"Intercept: {deflection_model.intercept_:.6f}")

# Predict for specific parameters: load_lbf = 80, thickness_mm = 1.24
print("\n" + "="*60)
print("PREDICTION FOR SPECIFIC PARAMETERS")
print("="*60)
input_params = np.array([[80, 1.24]])
predicted_stress = stress_model.predict(input_params)[0]
predicted_deflection = deflection_model.predict(input_params)[0]

print(f"Input Parameters: load_lbf=80 lbf, thickness_mm=1.24 mm")
print(f"Predicted max_stress_ksi: {predicted_stress:.4f}")
print(f"Predicted max_deflection_inch: {predicted_deflection:.4f}")

# Compare with actual data for the same parameters
actual_data = df[(df['load_lbf'] == 80) & (df['thickness_mm'] == 1.24)]
if not actual_data.empty:
    print("\nActual values from dataset:")
    print(f"Actual max_stress_ksi: {actual_data['max_stress_ksi'].values[0]:.4f}")
    print(f"Actual max_deflection_inch: {actual_data['max_deflection_inch'].values[0]:.4f}")

# Visualization
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Stress plot
axes[0].scatter(y_stress_test, y_stress_pred, alpha=0.6, color='blue')
axes[0].plot([y_stress_test.min(), y_stress_test.max()], [y_stress_test.min(), y_stress_test.max()], 'r--', lw=2)
axes[0].set_xlabel('Actual Stress (ksi)')
axes[0].set_ylabel('Predicted Stress (ksi)')
axes[0].set_title(f'Stress Prediction (R² = {r2_score(y_stress_test, y_stress_pred):.4f})')
axes[0].grid(True, alpha=0.3)

# Deflection plot
axes[1].scatter(y_deflection_test, y_deflection_pred, alpha=0.6, color='green')
axes[1].plot([y_deflection_test.min(), y_deflection_test.max()], [y_deflection_test.min(), y_deflection_test.max()], 'r--', lw=2)
axes[1].set_xlabel('Actual Deflection (inch)')
axes[1].set_ylabel('Predicted Deflection (inch)')
axes[1].set_title(f'Deflection Prediction (R² = {r2_score(y_deflection_test, y_deflection_pred):.4f})')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('prediction_results.png', dpi=100, bbox_inches='tight')
print("\n✓ Visualization saved as 'prediction_results.png'")
plt.show()
