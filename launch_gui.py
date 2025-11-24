"""
Digital Twin Pushover Analysis GUI Launcher
==========================================

This script launches the GUI application for Digital Twin Pushover Analysis of RC Walls.

Requirements:
- Python 3.7+
- Required packages: tkinter, numpy, pandas, matplotlib, scikit-learn, scipy

Usage:
- Double-click this file or run: python launch_gui.py
- The GUI will open with tabs for data loading, prediction, visualization, and performance analysis

Features:
- Load Excel data with wall parameters and pushover curves
- Train Random Forest models for bilinear parameter prediction
- Predict pushover curves for new walls
- Visualize results with interactive plots
- Export predictions to CSV and save plots
- View model performance metrics and feature importance

Author: Digital Twin Framework
Date: October 2025
"""

import sys
import os

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        'tkinter', 'numpy', 'pandas', 'matplotlib', 
        'sklearn', 'scipy'
    ]
    
    missing_packages = []
    
    for package in required_packages:
        try:
            if package == 'tkinter':
                import tkinter
            elif package == 'sklearn':
                import sklearn
            else:
                __import__(package)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("Missing required packages:")
        for pkg in missing_packages:
            print(f"  - {pkg}")
        print("\nPlease install missing packages using:")
        print("pip install numpy pandas matplotlib scikit-learn scipy")
        return False
    
    return True

def main():
    """Launch the Digital Twin GUI application"""
    print("=" * 60)
    print("Digital Twin Pushover Analysis - RC Walls")
    print("=" * 60)
    print("Checking requirements...")
    
    if not check_requirements():
        input("Press Enter to exit...")
        return
    
    print("All requirements satisfied!")
    print("Launching GUI application...")
    print("=" * 60)
    
    try:
        # Import and run the GUI
        from digital_twin_gui import DigitalTwinGUI
        import tkinter as tk
        
        root = tk.Tk()
        app = DigitalTwinGUI(root)
        
        print("GUI launched successfully!")
        print("You can now:")
        print("1. Load your Excel data file")
        print("2. Train the machine learning models")
        print("3. Predict pushover curves for new walls")
        print("4. Visualize and export results")
        print("=" * 60)
        
        root.mainloop()
        
    except Exception as e:
        print(f"Error launching GUI: {str(e)}")
        print("Please ensure all files are in the same directory:")
        print("- launch_gui.py")
        print("- digital_twin_gui.py")
        print("- digital_twin_pushover_data.xlsx")
        input("Press Enter to exit...")

if __name__ == "__main__":
    main()