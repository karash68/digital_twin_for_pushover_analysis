import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

# Ensure 'results' folder exists
os.makedirs('results', exist_ok=True)

# --- 1. Read Excel Data ---
excel_file = 'lhs_pushover_data_200walls.xlsx'  # Change to your actual file name

# Sheet 1: Wall/material parameters (30 samples)
df_params = pd.read_excel(excel_file, sheet_name='Sheet1')
X_train = df_params[['H', 'Lw', 't', 'fc', 'fy', 'rho_v', 'rho_h', 'axial_load', 'aspect_ratio','opening_ratio']].values

# Sheet 2: Experimental pushover curve (with Wall_ID for each wall)
df_curve = pd.read_excel(excel_file, sheet_name='Sheet2')
wall_ids = df_curve['Wall_ID'].unique()

# --- 2. Fit Experimental Curve to Extract Parameters for Each Wall ---
def bilinear_model(x, K0, deltay, alpha):
    y = np.where(x <= deltay, K0 * x, K0 * deltay + alpha * (x - deltay))
    return y

Y_train = []
all_pred_curves = []  # Initialize the list to collect all predicted curves
for wid in wall_ids:
    wall_curve = df_curve[df_curve['Wall_ID'] == wid]
    disp = wall_curve['Displacement_mm'].values
    force = wall_curve['BaseForce_kN'].values
    popt, _ = curve_fit(bilinear_model, disp, force, p0=[force[1]/disp[1], 10, 100])
    Y_train.append(popt)
Y_train = np.array(Y_train)  # Shape: (number_of_walls, 3)

# --- 3. Train Regression Model (Random Forest) ---
rf_K0 = RandomForestRegressor()
rf_deltay = RandomForestRegressor()
rf_alpha = RandomForestRegressor()

rf_K0.fit(X_train, Y_train[:, 0])
rf_deltay.fit(X_train, Y_train[:, 1])
rf_alpha.fit(X_train, Y_train[:, 2])

# --- 4. Loop Through All Walls and Plot Curves ---
for i, wid in enumerate(wall_ids):
    wall_curve = df_curve[df_curve['Wall_ID'] == wid]
    disp = wall_curve['Displacement_mm'].values
    force = wall_curve['BaseForce_kN'].values
    delta_range = np.linspace(disp.min(), disp.max(), 200)
    exp_interp = interp1d(disp, force, kind='cubic')
    base_force_smooth = exp_interp(delta_range)
    K0_exp, deltay_exp, alpha_exp = Y_train[i]
    V_exp_fit = bilinear_model(delta_range, K0_exp, deltay_exp, alpha_exp)
    X_new = X_train[i].reshape(1, -1)
    K0_pred = rf_K0.predict(X_new)[0]
    deltay_pred = rf_deltay.predict(X_new)[0]
    alpha_pred = rf_alpha.predict(X_new)[0]
    V_pred = bilinear_model(delta_range, K0_pred, deltay_pred, alpha_pred)
    
    # Save all three curves to CSV for Excel plotting
    df_all_curves = pd.DataFrame({
        'Displacement_mm': delta_range,
        'Experimental_Smooth_kN': base_force_smooth,
        'Fitted_Experimental_kN': V_exp_fit,
        'Digital_Twin_Prediction_kN': V_pred
    })
    df_all_curves.to_csv(f'results/digital_twin_all_curves_wall_{wid}.csv', index=False)
    
    # Plot and save
    plt.figure(figsize=(8, 5))
    plt.plot(delta_range, base_force_smooth, 'k-', label='Experimental (Smooth)')
    plt.plot(delta_range, V_exp_fit, 'r--', label='Fitted Experimental')
    plt.plot(delta_range, V_pred, 'b-.', label='Digital Twin Prediction')
    plt.xlabel('Displacement (mm)')
    plt.ylabel('Base Force (kN)')
    plt.title(f'Pushover Curve Comparison (Wall {wid})')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f'results/pushover_curve_wall_{wid}.png', dpi=300)
    plt.close()

# --- 5. Model Performance (R²) ---
models = ['Random Forest K0', 'Random Forest deltay', 'Random Forest alpha']
scores = [
    rf_K0.score(X_train, Y_train[:, 0]),
    rf_deltay.score(X_train, Y_train[:, 1]),
    rf_alpha.score(X_train, Y_train[:, 2])
]

plt.figure(figsize=(6, 4))
plt.bar(models, scores, alpha=0.7)
plt.ylabel('R² Score')
plt.title('Model Performance (R²)')
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/model_performance_r2.png', dpi=300)
plt.close()

# --- 6. Training Data Distribution ---
plt.figure(figsize=(6, 4))
plt.scatter(df_params['aspect_ratio'], df_params['fc'], c='blue', s=80, alpha=0.7)
plt.xlabel('Aspect Ratio (H/Lw)')
plt.ylabel('Concrete Strength (MPa)')
plt.title('Training Data Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('results/training_data_distribution.png', dpi=300)
plt.close()

# --- General Comparison Plot: All Predicted, Fitted, and Experimental Curves ---
plt.figure(figsize=(10, 6))

for i, wid in enumerate(wall_ids):
    wall_curve = df_curve[df_curve['Wall_ID'] == wid]
    disp = wall_curve['Displacement_mm'].values
    force = wall_curve['BaseForce_kN'].values
    delta_range = np.linspace(disp.min(), disp.max(), 200)
    exp_interp = interp1d(disp, force, kind='cubic')
    base_force_smooth = exp_interp(delta_range)
    K0_exp, deltay_exp, alpha_exp = Y_train[i]
    V_exp_fit = bilinear_model(delta_range, K0_exp, deltay_exp, alpha_exp)
    K0_pred = rf_K0.predict(X_train[i].reshape(1, -1))[0]
    deltay_pred = rf_deltay.predict(X_train[i].reshape(1, -1))[0]
    alpha_pred = rf_alpha.predict(X_train[i].reshape(1, -1))[0]
    V_pred = bilinear_model(delta_range, K0_pred, deltay_pred, alpha_pred)
    plt.plot(delta_range, base_force_smooth, '--', label=f'Wall {wid} Experimental (Smooth)', alpha=0.5)
    plt.plot(delta_range, V_exp_fit, ':', label=f'Wall {wid} Fitted Experimental', alpha=0.5)
    plt.plot(delta_range, V_pred, label=f'Wall {wid} Prediction', alpha=0.7)

for i, wid in enumerate(wall_ids):
    wall_curve = df_curve[df_curve['Wall_ID'] == wid]
    disp = wall_curve['Displacement_mm'].values
    force = wall_curve['BaseForce_kN'].values
    delta_range = np.linspace(disp.min(), disp.max(), 200)
    exp_interp = interp1d(disp, force, kind='cubic')
    base_force_smooth = exp_interp(delta_range)
    K0_exp, deltay_exp, alpha_exp = Y_train[i]
    V_exp_fit = bilinear_model(delta_range, K0_exp, deltay_exp, alpha_exp)
    K0_pred = rf_K0.predict(X_train[i].reshape(1, -1))[0]
    deltay_pred = rf_deltay.predict(X_train[i].reshape(1, -1))[0]
    alpha_pred = rf_alpha.predict(X_train[i].reshape(1, -1))[0]
    V_pred = bilinear_model(delta_range, K0_pred, deltay_pred, alpha_pred)
    plt.plot(delta_range, base_force_smooth, '--', label=f'Wall {wid} Experimental (Smooth)', alpha=0.5)
    plt.plot(delta_range, V_exp_fit, ':', label=f'Wall {wid} Fitted Experimental', alpha=0.5)
    plt.plot(delta_range, V_pred, label=f'Wall {wid} Prediction', alpha=0.7)
    all_pred_curves.append(V_pred)  # <-- Add this line to collect all predicted curves

# --- Add mean predicted curve ---
mean_pred_curve = np.mean(all_pred_curves, axis=0)
plt.plot(delta_range, mean_pred_curve, 'k-', linewidth=3, label='Mean Predicted Curve')

plt.xlabel('Displacement (mm)')
plt.ylabel('Base Force (kN)')
plt.title('Digital Twin Pushover Curves: Experimental, Fitted, Predicted & Mean (All Walls)')
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig('results/pushover_curve_all_walls.png', dpi=300)
plt.show()
plt.xlabel('Displacement (mm)')
plt.ylabel('Base Force (kN)')
plt.title('Digital Twin Pushover Curves: Experimental, Fitted, and Predicted (All Walls)')
plt.legend(fontsize=8)
plt.grid(True)
plt.tight_layout()
plt.savefig('results/pushover_curve_all_walls.png', dpi=300)
plt.show()

# --- Predict for a New Wall (not in training data) ---
# Replace these with your new wall's values:
H_new = 2.6
Lw_new = 1.25
t_new = 0.125
fc_new = 35
fy_new = 235
rho_v_new = 4
rho_h_new = 4
axial_load_new = 50
aspect_ratio_new = H_new / Lw_new
opening_ratio_new = 2.1 # Replace with the actual opening ratio for the new wall

X_new = np.array([[H_new, Lw_new, t_new, fc_new, fy_new, rho_v_new, rho_h_new, axial_load_new, aspect_ratio_new, opening_ratio_new]])

K0_pred = rf_K0.predict(X_new)[0]
K0_pred = max(rf_K0.predict(X_new)[0], 0)
K0_pred = rf_K0.predict(X_new)[0]
deltay_pred = rf_deltay.predict(X_new)[0]
alpha_pred = rf_alpha.predict(X_new)[0]

delta_range_new = np.linspace(0, deltay_pred * 3, 200)
V_pred_new = bilinear_model(delta_range_new, K0_pred, deltay_pred, alpha_pred)

poly_coeffs = np.polyfit(delta_range_new, V_pred_new, 4)
delta_range_new_fine = np.linspace(0, deltay_pred * 3, 1000)
V_pred_new_poly4 = np.polyval(poly_coeffs, delta_range_new_fine)

# Save to CSV for Excel plotting

# Save bilinear predicted curve
df_pred_new_bilinear = pd.DataFrame({
    'Displacement_mm': delta_range_new,
    'Digital_Twin_Prediction_kN': V_pred_new
})
df_pred_new_bilinear.to_csv('results/digital_twin_predicted_curve_new_wall_bilinear.csv', index=False)

# Save curved predicted curve
df_pred_new_poly4 = pd.DataFrame({
    'Displacement_mm': delta_range_new_fine,
    'Digital_Twin_Prediction_Poly4_kN': V_pred_new_poly4
})
df_pred_new_poly4.to_csv('results/digital_twin_predicted_curve_new_wall_poly4.csv', index=False)

# Plot the predicted curve
plt.figure(figsize=(8, 5))
plt.plot(delta_range_new, V_pred_new, 'b-', label='Digital Twin Prediction (Bilinear)')
plt.plot(delta_range_new_fine, V_pred_new_poly4, 'g-', label='Digital Twin Prediction (4th-degree Poly)')
plt.xlabel('Displacement (mm)')
plt.ylabel('Base Force (kN)')
plt.title('Predicted Pushover Curve for New Wall')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/pushover_curve_new_wall.png', dpi=300)
plt.show()