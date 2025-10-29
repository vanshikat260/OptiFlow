# OptiFlow
Logistics Network Optimization: Predictive Performance Dashboard
This project implements a Streamlit dashboard that analyzes logistics data, calculates profitability, monitors delivery performance, and integrates a Machine Learning model for predictive optimization.
The goal is to move from reactive management to proactive intervention by flagging high-risk orders before they turn into actual delays.

💾 Project Structure
To run the application, you must organize your files exactly as shown below:
/Project_Root
├── streamlit_logistics_dashboard.py <-- The main application file
├── requirements.txt <-- List of dependencies
├── README.md <-- This instruction file
└── /data/ <-- MANDATORY FOLDER
  ├── orders.csv
  ├── delivery_performance.csv
  ├── cost_breakdown.csv
  ├── routes_distance.csv
  ├── customer_feedback.csv
  ├── vehicle_fleet.csv
  └── warehouse_inventory.csv

Action: Ensure you have created the data folder and placed all seven extracted CSV files inside it.
🚀 Running the Streamlit App (VS Code)
This is the standard and most reliable way to execute the application locally.
Step 1: Install Dependencies
Open your VS Code terminal and run the following command to install all necessary Python packages:
pip install -r requirements.txt

Step 2: Run the Application
Once the packages are installed, execute the following command in the same VS Code terminal:
streamlit run streamlit_logistics_dashboard.py

Expected Outcome
Your default web browser should automatically open to the application address (usually http://localhost:8501).
The application will display the main dashboard with two tabs: Performance & Profit Analysis and Optimization Prototype (ML).
The sidebar allows filtering by Customer Segment for interactive analysis.
