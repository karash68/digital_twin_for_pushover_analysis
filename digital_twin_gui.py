import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.ensemble import RandomForestRegressor
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d
import os

class DigitalTwinGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Digital Twin Pushover Analysis - RC Walls")
        self.root.geometry("1200x800")
        
        # Initialize variables
        self.df_params = None
        self.df_curve = None
        self.X_train = None
        self.Y_train = None
        self.rf_K0 = None
        self.rf_deltay = None
        self.rf_alpha = None
        self.wall_ids = None
        self.excel_file = None
        
        self.setup_gui()
        
    def setup_gui(self):
        # Create main notebook for tabs
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(fill='both', expand=True, padx=10, pady=10)
        
        # Tab 1: Data Loading and Training
        self.tab1 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab1, text="Data & Training")
        self.setup_tab1()
        
        # Tab 2: New Wall Prediction
        self.tab2 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab2, text="New Wall Prediction")
        self.setup_tab2()
        
        # Tab 3: Results Visualization
        self.tab3 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab3, text="Results & Visualization")
        self.setup_tab3()
        
        # Tab 4: Model Performance
        self.tab4 = ttk.Frame(self.notebook)
        self.notebook.add(self.tab4, text="Model Performance")
        self.setup_tab4()
        
    def setup_tab1(self):
        # File selection frame
        file_frame = ttk.LabelFrame(self.tab1, text="Data Loading", padding=10)
        file_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Label(file_frame, text="Excel File:").grid(row=0, column=0, sticky='w')
        self.file_var = tk.StringVar()
        ttk.Entry(file_frame, textvariable=self.file_var, width=50).grid(row=0, column=1, padx=5)
        ttk.Button(file_frame, text="Browse", command=self.browse_file).grid(row=0, column=2, padx=5)
        ttk.Button(file_frame, text="Load Data", command=self.load_data).grid(row=0, column=3, padx=5)
        
        # Data info frame
        info_frame = ttk.LabelFrame(self.tab1, text="Data Information", padding=10)
        info_frame.pack(fill='x', padx=10, pady=5)
        
        self.data_info = tk.Text(info_frame, height=8, width=80)
        self.data_info.pack(fill='both', expand=True)
        
        # Training frame
        train_frame = ttk.LabelFrame(self.tab1, text="Model Training", padding=10)
        train_frame.pack(fill='x', padx=10, pady=5)
        
        ttk.Button(train_frame, text="Train Models", command=self.train_models, style='Accent.TButton').pack(pady=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(train_frame, mode='indeterminate')
        self.progress.pack(fill='x', pady=5)
        
        # Status label
        self.status_label = ttk.Label(train_frame, text="Ready to load data...")
        self.status_label.pack(pady=5)
        
    def setup_tab2(self):
        # Input parameters frame
        params_frame = ttk.LabelFrame(self.tab2, text="New Wall Parameters", padding=10)
        params_frame.pack(fill='x', padx=10, pady=5)
        
        # Create input fields
        self.param_vars = {}
        params = [
            ("Height (H) [m]", "H", 2.6),
            ("Length (Lw) [m]", "Lw", 1.25),
            ("Thickness (t) [m]", "t", 0.125),
            ("Concrete Strength (fc) [MPa]", "fc", 35),
            ("Steel Yield Strength (fy) [MPa]", "fy", 235),
            ("Vertical Rebar Ratio (ρv) [%]", "rho_v", 4),
            ("Horizontal Rebar Ratio (ρh) [%]", "rho_h", 4),
            ("Axial Load [kN]", "axial_load", 50),
            ("Opening Ratio [%]", "opening_ratio", 2.1)
        ]
        
        for i, (label, key, default) in enumerate(params):
            row, col = divmod(i, 3)
            ttk.Label(params_frame, text=label).grid(row=row*2, column=col, sticky='w', padx=5, pady=2)
            var = tk.DoubleVar(value=default)
            self.param_vars[key] = var
            ttk.Entry(params_frame, textvariable=var, width=15).grid(row=row*2+1, column=col, padx=5, pady=2)
        
        # Prediction frame
        pred_frame = ttk.LabelFrame(self.tab2, text="Prediction", padding=10)
        pred_frame.pack(fill='both', expand=True, padx=10, pady=5)
        
        ttk.Button(pred_frame, text="Predict Pushover Curve", command=self.predict_new_wall, 
                  style='Accent.TButton').pack(pady=10)
        
        # Results display
        self.pred_results = tk.Text(pred_frame, height=6, width=80)
        self.pred_results.pack(fill='x', pady=5)
        
        # Export buttons
        export_frame = ttk.Frame(pred_frame)
        export_frame.pack(fill='x', pady=5)
        
        ttk.Button(export_frame, text="Export Bilinear CSV", 
                  command=self.export_bilinear_csv).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Export Polynomial CSV", 
                  command=self.export_poly_csv).pack(side='left', padx=5)
        ttk.Button(export_frame, text="Save Plot", 
                  command=self.save_prediction_plot).pack(side='left', padx=5)
        
    def setup_tab3(self):
        # Visualization controls
        viz_frame = ttk.LabelFrame(self.tab3, text="Visualization Controls", padding=10)
        viz_frame.pack(fill='x', padx=10, pady=5)
        
        # Wall selection
        ttk.Label(viz_frame, text="Select Wall:").grid(row=0, column=0, sticky='w')
        self.wall_var = tk.StringVar()
        self.wall_combo = ttk.Combobox(viz_frame, textvariable=self.wall_var, width=15)
        self.wall_combo.grid(row=0, column=1, padx=5)
        
        ttk.Button(viz_frame, text="Plot Individual Wall", 
                  command=self.plot_individual_wall).grid(row=0, column=2, padx=5)
        ttk.Button(viz_frame, text="Plot All Walls", 
                  command=self.plot_all_walls).grid(row=0, column=3, padx=5)
        ttk.Button(viz_frame, text="Generate All Figures", 
                  command=self.generate_all_figures).grid(row=0, column=4, padx=5)
        
        # Matplotlib canvas
        self.fig = Figure(figsize=(12, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, self.tab3)
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
    def setup_tab4(self):
        # Performance metrics frame
        perf_frame = ttk.LabelFrame(self.tab4, text="Model Performance Metrics", padding=10)
        perf_frame.pack(fill='x', padx=10, pady=5)
        
        self.perf_text = tk.Text(perf_frame, height=10, width=80)
        self.perf_text.pack(fill='both', expand=True)
        
        # Performance visualization
        self.perf_fig = Figure(figsize=(12, 6))
        self.perf_canvas = FigureCanvasTkAgg(self.perf_fig, self.tab4)
        self.perf_canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=5)
        
    def browse_file(self):
        filename = filedialog.askopenfilename(
            title="Select Excel File",
            filetypes=[("Excel files", "*.xlsx *.xls")]
        )
        if filename:
            self.file_var.set(filename)
    
    def load_data(self):
        if not self.file_var.get():
            messagebox.showerror("Error", "Please select an Excel file first.")
            return
        
        try:
            self.excel_file = self.file_var.get()
            
            # Load data
            self.df_params = pd.read_excel(self.excel_file, sheet_name='Sheet1')
            self.df_curve = pd.read_excel(self.excel_file, sheet_name='Sheet2')
            
            # Prepare training data
            self.X_train = self.df_params[['H', 'Lw', 't', 'fc', 'fy', 'rho_v', 'rho_h', 
                                         'axial_load', 'aspect_ratio', 'opening_ratio']].values
            self.wall_ids = self.df_curve['Wall_ID'].unique()
            
            # Update wall selection combo
            self.wall_combo['values'] = [f"Wall {wid}" for wid in self.wall_ids]
            
            # Display data info
            info_text = f"""Data Loaded Successfully!
            
Number of walls: {len(self.wall_ids)}
Wall IDs: {list(self.wall_ids)}

Parameters shape: {self.X_train.shape}
Curve data points: {len(self.df_curve)}

Parameter columns: {list(self.df_params.columns)}
Curve columns: {list(self.df_curve.columns)}

Ready for model training."""
            
            self.data_info.delete(1.0, tk.END)
            self.data_info.insert(tk.END, info_text)
            self.status_label.config(text="Data loaded successfully! Ready to train models.")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load data: {str(e)}")
    
    def bilinear_model(self, x, K0, deltay, alpha):
        y = np.where(x <= deltay, K0 * x, K0 * deltay + alpha * (x - deltay))
        return y
    
    def train_models(self):
        if self.X_train is None:
            messagebox.showerror("Error", "Please load data first.")
            return
        
        try:
            self.status_label.config(text="Training models...")
            self.progress.start()
            self.root.update()
            
            # Fit bilinear model to extract parameters
            self.Y_train = []
            for wid in self.wall_ids:
                wall_curve = self.df_curve[self.df_curve['Wall_ID'] == wid]
                disp = wall_curve['Displacement_mm'].values
                force = wall_curve['BaseForce_kN'].values
                popt, _ = curve_fit(self.bilinear_model, disp, force, p0=[force[1]/disp[1], 10, 100])
                self.Y_train.append(popt)
            self.Y_train = np.array(self.Y_train)
            
            # Train Random Forest models
            self.rf_K0 = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rf_deltay = RandomForestRegressor(n_estimators=100, random_state=42)
            self.rf_alpha = RandomForestRegressor(n_estimators=100, random_state=42)
            
            self.rf_K0.fit(self.X_train, self.Y_train[:, 0])
            self.rf_deltay.fit(self.X_train, self.Y_train[:, 1])
            self.rf_alpha.fit(self.X_train, self.Y_train[:, 2])
            
            self.progress.stop()
            self.status_label.config(text="Models trained successfully!")
            
            # Update performance metrics
            self.update_performance_metrics()
            
            messagebox.showinfo("Success", "Models trained successfully!")
            
        except Exception as e:
            self.progress.stop()
            self.status_label.config(text="Training failed!")
            messagebox.showerror("Error", f"Failed to train models: {str(e)}")
    
    def update_performance_metrics(self):
        if self.rf_K0 is None:
            return
        
        # Calculate R² scores
        r2_K0 = self.rf_K0.score(self.X_train, self.Y_train[:, 0])
        r2_deltay = self.rf_deltay.score(self.X_train, self.Y_train[:, 1])
        r2_alpha = self.rf_alpha.score(self.X_train, self.Y_train[:, 2])
        
        # Feature importance
        feature_names = ['H', 'Lw', 't', 'fc', 'fy', 'rho_v', 'rho_h', 'axial_load', 'aspect_ratio', 'opening_ratio']
        
        perf_text = f"""Model Performance Metrics:

R² Scores:
- Initial Stiffness (K0): {r2_K0:.4f}
- Yield Displacement (δy): {r2_deltay:.4f}
- Post-yield Stiffness (α): {r2_alpha:.4f}

Feature Importance (K0):
"""
        for i, importance in enumerate(self.rf_K0.feature_importances_):
            perf_text += f"- {feature_names[i]}: {importance:.4f}\n"
        
        self.perf_text.delete(1.0, tk.END)
        self.perf_text.insert(tk.END, perf_text)
        
        # Plot performance
        self.perf_fig.clear()
        
        # R² scores plot
        ax1 = self.perf_fig.add_subplot(1, 2, 1)
        models = ['K0', 'δy', 'α']
        scores = [r2_K0, r2_deltay, r2_alpha]
        ax1.bar(models, scores, alpha=0.7, color=['blue', 'green', 'red'])
        ax1.set_ylabel('R² Score')
        ax1.set_title('Model Performance')
        ax1.set_ylim(0, 1)
        ax1.grid(True, alpha=0.3)
        
        # Feature importance plot
        ax2 = self.perf_fig.add_subplot(1, 2, 2)
        importance_indices = np.argsort(self.rf_K0.feature_importances_)[::-1][:5]
        top_features = [feature_names[i] for i in importance_indices]
        top_importance = [self.rf_K0.feature_importances_[i] for i in importance_indices]
        
        ax2.barh(top_features[::-1], top_importance[::-1], alpha=0.7)
        ax2.set_xlabel('Feature Importance')
        ax2.set_title('Top 5 Features (K0)')
        
        self.perf_fig.tight_layout()
        self.perf_canvas.draw()
    
    def predict_new_wall(self):
        if self.rf_K0 is None:
            messagebox.showerror("Error", "Please train models first.")
            return
        
        try:
            # Get input parameters
            params = {}
            for key, var in self.param_vars.items():
                params[key] = var.get()
            
            # Calculate aspect ratio
            aspect_ratio = params['H'] / params['Lw']
            
            # Prepare input array
            X_new = np.array([[params['H'], params['Lw'], params['t'], params['fc'], 
                             params['fy'], params['rho_v'], params['rho_h'], 
                             params['axial_load'], aspect_ratio, params['opening_ratio']]])
            
            # Predict parameters
            K0_pred = self.rf_K0.predict(X_new)[0]
            deltay_pred = self.rf_deltay.predict(X_new)[0]
            alpha_pred = self.rf_alpha.predict(X_new)[0]
            
            # Generate predicted curve
            self.delta_range_new = np.linspace(0, deltay_pred * 3, 200)
            self.V_pred_new = self.bilinear_model(self.delta_range_new, K0_pred, deltay_pred, alpha_pred)
            
            # Generate polynomial curve
            poly_coeffs = np.polyfit(self.delta_range_new, self.V_pred_new, 4)
            self.delta_range_new_fine = np.linspace(0, deltay_pred * 3, 1000)
            self.V_pred_new_poly4 = np.polyval(poly_coeffs, self.delta_range_new_fine)
            
            # Display results
            results_text = f"""Prediction Results:

Input Parameters:
- Height: {params['H']:.2f} m
- Length: {params['Lw']:.2f} m  
- Thickness: {params['t']:.3f} m
- Concrete Strength: {params['fc']:.1f} MPa
- Steel Strength: {params['fy']:.1f} MPa
- Aspect Ratio: {aspect_ratio:.2f}

Predicted Bilinear Parameters:
- Initial Stiffness (K0): {K0_pred:.2f} kN/mm
- Yield Displacement (δy): {deltay_pred:.2f} mm
- Post-yield Stiffness (α): {alpha_pred:.2f} kN/mm
- Maximum Force: {K0_pred * deltay_pred:.2f} kN
"""
            
            self.pred_results.delete(1.0, tk.END)
            self.pred_results.insert(tk.END, results_text)
            
            # Plot prediction
            self.plot_prediction()
            
        except Exception as e:
            messagebox.showerror("Error", f"Prediction failed: {str(e)}")
    
    def plot_prediction(self):
        # Switch to results tab and plot
        self.notebook.select(self.tab3)
        
        self.fig.clear()
        ax = self.fig.add_subplot(1, 1, 1)
        
        ax.plot(self.delta_range_new, self.V_pred_new, 'b-', linewidth=2, 
                label='Digital Twin Prediction (Bilinear)')
        ax.plot(self.delta_range_new_fine, self.V_pred_new_poly4, 'g--', linewidth=2, 
                label='Digital Twin Prediction (4th-degree Poly)')
        
        ax.set_xlabel('Displacement (mm)')
        ax.set_ylabel('Base Force (kN)')
        ax.set_title('Predicted Pushover Curve for New Wall')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        self.fig.tight_layout()
        self.canvas.draw()
    
    def plot_individual_wall(self):
        if not self.wall_var.get() or self.rf_K0 is None:
            messagebox.showerror("Error", "Please select a wall and ensure models are trained.")
            return
        
        try:
            wall_id = int(self.wall_var.get().split()[-1])
            wall_idx = list(self.wall_ids).index(wall_id)
            
            # Get experimental data
            wall_curve = self.df_curve[self.df_curve['Wall_ID'] == wall_id]
            disp = wall_curve['Displacement_mm'].values
            force = wall_curve['BaseForce_kN'].values
            
            # Generate smooth experimental curve
            delta_range = np.linspace(disp.min(), disp.max(), 200)
            exp_interp = interp1d(disp, force, kind='cubic')
            base_force_smooth = exp_interp(delta_range)
            
            # Get fitted experimental curve
            K0_exp, deltay_exp, alpha_exp = self.Y_train[wall_idx]
            V_exp_fit = self.bilinear_model(delta_range, K0_exp, deltay_exp, alpha_exp)
            
            # Get predicted curve
            X_wall = self.X_train[wall_idx].reshape(1, -1)
            K0_pred = self.rf_K0.predict(X_wall)[0]
            deltay_pred = self.rf_deltay.predict(X_wall)[0]
            alpha_pred = self.rf_alpha.predict(X_wall)[0]
            V_pred = self.bilinear_model(delta_range, K0_pred, deltay_pred, alpha_pred)
            
            # Plot
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            
            ax.plot(delta_range, base_force_smooth, 'k-', linewidth=2, label='Experimental (Smooth)')
            ax.plot(delta_range, V_exp_fit, 'r--', linewidth=2, label='Fitted Experimental')
            ax.plot(delta_range, V_pred, 'b-.', linewidth=2, label='Digital Twin Prediction')
            
            ax.set_xlabel('Displacement (mm)')
            ax.set_ylabel('Base Force (kN)')
            ax.set_title(f'Pushover Curve Comparison (Wall {wall_id})')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot wall: {str(e)}")
    
    def plot_all_walls(self):
        if self.rf_K0 is None:
            messagebox.showerror("Error", "Please train models first.")
            return
        
        try:
            self.fig.clear()
            ax = self.fig.add_subplot(1, 1, 1)
            
            all_pred_curves = []
            
            for i, wid in enumerate(self.wall_ids):
                wall_curve = self.df_curve[self.df_curve['Wall_ID'] == wid]
                disp = wall_curve['Displacement_mm'].values
                force = wall_curve['BaseForce_kN'].values
                
                delta_range = np.linspace(disp.min(), disp.max(), 200)
                exp_interp = interp1d(disp, force, kind='cubic')
                base_force_smooth = exp_interp(delta_range)
                
                K0_exp, deltay_exp, alpha_exp = self.Y_train[i]
                V_exp_fit = self.bilinear_model(delta_range, K0_exp, deltay_exp, alpha_exp)
                
                K0_pred = self.rf_K0.predict(self.X_train[i].reshape(1, -1))[0]
                deltay_pred = self.rf_deltay.predict(self.X_train[i].reshape(1, -1))[0]
                alpha_pred = self.rf_alpha.predict(self.X_train[i].reshape(1, -1))[0]
                V_pred = self.bilinear_model(delta_range, K0_pred, deltay_pred, alpha_pred)
                
                ax.plot(delta_range, base_force_smooth, '--', alpha=0.5, color='gray')
                ax.plot(delta_range, V_pred, '-', alpha=0.7, label=f'Wall {wid}' if i < 10 else "")
                all_pred_curves.append(V_pred)
            
            # Plot mean curve
            if all_pred_curves:
                mean_pred_curve = np.mean(all_pred_curves, axis=0)
                ax.plot(delta_range, mean_pred_curve, 'k-', linewidth=3, label='Mean Predicted Curve')
            
            ax.set_xlabel('Displacement (mm)')
            ax.set_ylabel('Base Force (kN)')
            ax.set_title('Digital Twin Pushover Curves: All Walls')
            ax.legend(fontsize=8, ncol=2)
            ax.grid(True, alpha=0.3)
            
            self.fig.tight_layout()
            self.canvas.draw()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to plot all walls: {str(e)}")
    
    def generate_all_figures(self):
        if self.rf_K0 is None:
            messagebox.showerror("Error", "Please train models first.")
            return
        
        try:
            # Generate individual wall plots
            for wid in self.wall_ids:
                self.wall_var.set(f"Wall {wid}")
                self.plot_individual_wall()
                
                # Save plot
                self.fig.savefig(f'pushover_curve_wall_{wid}.png', dpi=300, bbox_inches='tight')
            
            # Generate performance plot
            self.plot_performance_chart()
            
            # Generate data distribution plot
            self.plot_data_distribution()
            
            messagebox.showinfo("Success", "All figures generated and saved!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate figures: {str(e)}")
    
    def plot_performance_chart(self):
        if self.rf_K0 is None:
            return
        
        r2_K0 = self.rf_K0.score(self.X_train, self.Y_train[:, 0])
        r2_deltay = self.rf_deltay.score(self.X_train, self.Y_train[:, 1])
        r2_alpha = self.rf_alpha.score(self.X_train, self.Y_train[:, 2])
        
        plt.figure(figsize=(8, 6))
        models = ['Random Forest K0', 'Random Forest deltay', 'Random Forest alpha']
        scores = [r2_K0, r2_deltay, r2_alpha]
        plt.bar(models, scores, alpha=0.7, color=['blue', 'green', 'red'])
        plt.ylabel('R² Score')
        plt.title('Model Performance (R²)')
        plt.ylim(0, 1)
        plt.grid(True, alpha=0.3)
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig('model_performance_r2.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_data_distribution(self):
        if self.df_params is None:
            return
        
        plt.figure(figsize=(8, 6))
        plt.scatter(self.df_params['aspect_ratio'], self.df_params['fc'], 
                   c='blue', s=80, alpha=0.7, edgecolors='black')
        plt.xlabel('Aspect Ratio (H/Lw)')
        plt.ylabel('Concrete Strength (MPa)')
        plt.title('Training Data Distribution')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig('training_data_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    def export_bilinear_csv(self):
        if not hasattr(self, 'V_pred_new'):
            messagebox.showerror("Error", "Please make a prediction first.")
            return
        
        try:
            df_pred = pd.DataFrame({
                'Displacement_mm': self.delta_range_new,
                'Digital_Twin_Prediction_kN': self.V_pred_new
            })
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Bilinear Prediction"
            )
            if filename:
                df_pred.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Bilinear prediction saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def export_poly_csv(self):
        if not hasattr(self, 'V_pred_new_poly4'):
            messagebox.showerror("Error", "Please make a prediction first.")
            return
        
        try:
            df_pred = pd.DataFrame({
                'Displacement_mm': self.delta_range_new_fine,
                'Digital_Twin_Prediction_Poly4_kN': self.V_pred_new_poly4
            })
            filename = filedialog.asksaveasfilename(
                defaultextension=".csv",
                filetypes=[("CSV files", "*.csv")],
                title="Save Polynomial Prediction"
            )
            if filename:
                df_pred.to_csv(filename, index=False)
                messagebox.showinfo("Success", f"Polynomial prediction saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to export: {str(e)}")
    
    def save_prediction_plot(self):
        if not hasattr(self, 'V_pred_new'):
            messagebox.showerror("Error", "Please make a prediction first.")
            return
        
        try:
            filename = filedialog.asksaveasfilename(
                defaultextension=".png",
                filetypes=[("PNG files", "*.png"), ("PDF files", "*.pdf")],
                title="Save Prediction Plot"
            )
            if filename:
                self.fig.savefig(filename, dpi=300, bbox_inches='tight')
                messagebox.showinfo("Success", f"Plot saved to {filename}")
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save plot: {str(e)}")

def main():
    root = tk.Tk()
    app = DigitalTwinGUI(root)
    root.mainloop()

if __name__ == "__main__":
    main()