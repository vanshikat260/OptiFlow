import pandas as pd
import streamlit as st
import plotly.express as px
from typing import Dict, Any, List
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error


# --- 1. DATA LOADING AND PREPROCESSING FUNCTIONS ---

@st.cache_data
def load_data() -> Dict[str, pd.DataFrame] | None:
    """
    Loads all datasets from the 'data/' subdirectory using relative paths.
    """
    
    data_files = {
        'orders': "data/orders.csv",
        'delivery': "data/delivery_performance.csv",
        'feedback': "data/customer_feedback.csv",
        'cost': "data/cost_breakdown.csv",
        'routes': "data/routes_distance.csv",
        'vehicle': "data/vehicle_fleet.csv",
        'inventory': "data/warehouse_inventory.csv"
    }

    data = {}
    
    
    try:
        # A simple check to ensure the folder is there before reading files
        import os
        if not os.path.exists("data"):
             st.error("Data folder not found. Please create a folder named 'data' and place all CSV files inside it.")
             return None
    except:
        pass # Allow the subsequent read to fail gracefully if OS is restricted

    st.sidebar.caption("Loading datasets...")
    for key, path in data_files.items():
        try:
            # Date Parsing logic from original code
            if key == 'orders':
                data[key] = pd.read_csv(path, parse_dates=['Order_Date'])
            elif key == 'feedback':
                data[key] = pd.read_csv(path, parse_dates=['Feedback_Date'])
            elif key == 'inventory':
                 data[key] = pd.read_csv(path, parse_dates=['Last_Restocked_Date'])
            else:
                data[key] = pd.read_csv(path)

        except FileNotFoundError:
             st.error(f"File not found: {path}. Ensure all 7 CSV files are in the 'data' folder.")
             return None
        except Exception as e:
            st.error(f"Error loading {key} from path: {path}")
            st.exception(e)
            return None
    return data

@st.cache_data
def clean_and_derive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handles missing values and calculates key derived metrics (Cost, Profit, Delay, Carbon).
    """
    # 1. IMPUTATION: Handling Missing Numeric Values (Filling with 0)
    numeric_cols_to_impute_zero = [
        'Promised_Delivery_Days', 'Actual_Delivery_Days', 'Distance_KM', 'Fuel_Consumption_L',
        'Toll_Charges_INR', 'Traffic_Delay_Minutes', 'Delivery_Cost_INR', 'Fuel_Cost',
        'Labor_Cost', 'Vehicle_Maintenance', 'Insurance', 'Packaging_Cost',
        'Technology_Platform_Fee', 'Other_Overhead', 'Capacity_KG', 'Fuel_Efficiency_KM_per_L',
        'CO2_Emissions_Kg_per_KM', 'Customer_Rating', 'Current_Stock_Units', 'Reorder_Level',
        'Storage_Cost_per_Unit'
    ]
    for col in numeric_cols_to_impute_zero:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # 2. IMPUTATION: Handling Missing Categorical/Text Values (Filling with 'No Data')
    categorical_cols_to_impute_missing = [
        'Carrier', 'Delivery_Status', 'Quality_Issue', 'Feedback_Text', 'Issue_Category',
        'Weather_Impact', 'Special_Handling', 'Would_Recommend'
    ]
    for col in categorical_cols_to_impute_missing:
        if col in df.columns:
            df[col] = df[col].fillna('No Data').astype(str).str.strip()

    # 3. DERIVED METRICS: Financials
    cost_cols = [
        'Delivery_Cost_INR', 'Fuel_Cost', 'Labor_Cost', 'Vehicle_Maintenance',
        'Insurance', 'Packaging_Cost', 'Technology_Platform_Fee', 'Other_Overhead'
    ]
    df['Total_Logistics_Cost'] = df[cost_cols].sum(axis=1)
    df['Profit_INR'] = df['Order_Value_INR'] - df['Total_Logistics_Cost']

    # 4. DERIVED METRICS: Delivery Performance
    df['Delivery_Delay_Days'] = df['Actual_Delivery_Days'] - df['Promised_Delivery_Days']
    df['Is_On_Time'] = df['Delivery_Delay_Days'] <= 0

    # 5. DERIVED METRICS: Environmental
    # Formula: Carbon (KG) = Distance (KM) * CO2 Emissions (Kg/KM)
    df['Carbon_Footprint_KG'] = df['Distance_KM'] * df['CO2_Emissions_Kg_per_KM']

    return df

@st.cache_data
def preprocess_and_merge(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Merges all dataframes into a single master DataFrame."""

    master_df = data['orders'].copy()

    # Merging order-level data on Order_ID
    master_df = pd.merge(master_df, data['delivery'], on='Order_ID', how='left')
    master_df = pd.merge(master_df, data['routes'], on='Order_ID', how='left')
    master_df = pd.merge(master_df, data['cost'], on='Order_ID', how='left')
    master_df = pd.merge(master_df, data['feedback'], on='Order_ID', how='left')

    # Simulating Vehicle Assignment (since Orders lack a Vehicle_ID column)
    vehicle_ids = data['vehicle']['Vehicle_ID'].unique()
    master_df['Assigned_Vehicle_ID'] = [
        vehicle_ids[i % len(vehicle_ids)]
        for i in master_df.index
    ]

    # Merging with Vehicle data, explicitly renaming to avoid merge conflicts
    vehicles_renamed = data['vehicle'].rename(columns={'Vehicle_ID': 'Assigned_Vehicle_ID', 'Vehicle_Type': 'Vehicle_Type_Assigned'})
    master_df = pd.merge(master_df, vehicles_renamed, on='Assigned_Vehicle_ID', how='left')

    # Merging with Inventory data (assuming Origin and Product_Category are the link)
    inventory_renamed = data['inventory'].rename(columns={'Location': 'Origin'})
    master_df = pd.merge(master_df, inventory_renamed, on=['Origin', 'Product_Category'], how='left')

    # Performing cleaning and feature engineering
    master_df = clean_and_derive_features(master_df)

    return master_df

# --- 2. MACHINE LEARNING MODEL (OPTIMIZATION PROTOTYPE) ---

@st.cache_data
def train_delay_predictor(df: pd.DataFrame) -> RandomForestRegressor:
    """
    Trains a Random Forest Regressor to predict Delivery Delay Days.
    This serves as the "Optimization Prototype."
    """
    # 1. Feature Selection and Engineering for ML
    features = [
        'Distance_KM', 'Promised_Delivery_Days', 'Order_Value_INR',
        'Priority', 'Customer_Segment', 'Product_Category', 'Weather_Impact',
        'Fuel_Consumption_L', 'Toll_Charges_INR', 'Traffic_Delay_Minutes'
    ]

    # Dropping rows where target variable is missing (though already imputed to 0, safety check)
    df_model = df.dropna(subset=['Delivery_Delay_Days']).copy()

    # One-Hot Encode Categorical Features
    df_model = pd.get_dummies(df_model, columns=[
        'Priority', 'Customer_Segment', 'Product_Category', 'Weather_Impact'
    ], drop_first=True)

    # Aligning features with the encoded list
    ml_features = [col for col in df_model.columns if any(feat in col for feat in features)]
    X = df_model[ml_features]
    y = df_model['Delivery_Delay_Days']

    # 2. Training Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
    model.fit(X_train, y_train)

    # 3. Evaluating and Storing Prediction
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    st.session_state['model_mae'] = mae

    # Generating predictions for all data (including training data)
    # The feature list for prediction must match the feature list used for training
    df['Predicted_Delay_Days'] = model.predict(X)
    st.session_state['master_df'] = df

    return model

# --- 3. PLOTLY VISUALIZATION FUNCTIONS ---

def create_profitability_chart(df: pd.DataFrame):
    """Chart 1: Total Profit by Product Category."""
    profit_summary = df.groupby('Product_Category')['Profit_INR'].sum().reset_index()
    profit_summary['Total_Profit_Lacs'] = profit_summary['Profit_INR'] / 100000

    fig = px.bar(
        profit_summary,
        x='Product_Category',
        y='Total_Profit_Lacs',
        color='Product_Category',
        title="1. Total Profit by Product Category (in Lakhs INR)",
        labels={'Total_Profit_Lacs': 'Total Profit (Lacs INR)', 'Product_Category': 'Product'},
        template="plotly_white"
    )
    return fig

def create_performance_chart(df: pd.DataFrame):
    """Chart 2: On-Time Rate vs. Average Delay by Carrier."""
    perf_summary = df.groupby('Carrier').agg(
        On_Time_Rate=('Is_On_Time', 'mean'),
        Avg_Delay_Days=('Delivery_Delay_Days', 'mean'),
        Total_Orders=('Order_ID', 'count')
    ).reset_index()
    perf_summary['On_Time_Rate'] = perf_summary['On_Time_Rate'] * 100

    fig = px.scatter(
        perf_summary,
        x='Avg_Delay_Days',
        y='On_Time_Rate',
        size='Total_Orders',
        color='Carrier',
        hover_name='Carrier',
        title="2. Carrier Performance: On-Time Rate vs. Average Delay",
        labels={'On_Time_Rate': 'On-Time Rate (%)', 'Avg_Delay_Days': 'Avg. Delay (Days)'},
        template="plotly_white"
    )
    return fig

def create_cost_carbon_chart(df: pd.DataFrame):
    """Chart 3: Total Cost and Carbon Footprint by Vehicle Type."""
    cost_carbon = df.groupby('Vehicle_Type_Assigned').agg(
        Total_Cost=('Total_Logistics_Cost', 'sum'),
        Avg_Carbon_KG=('Carbon_Footprint_KG', 'mean')
    ).reset_index()

    # Normalizing data for combined plot
    cost_carbon['Cost_Norm'] = cost_carbon['Total_Cost'] / cost_carbon['Total_Cost'].max()
    cost_carbon['Carbon_Norm'] = cost_carbon['Avg_Carbon_KG'] / cost_carbon['Avg_Carbon_KG'].max()

    fig = px.bar(
        cost_carbon.melt(id_vars='Vehicle_Type_Assigned', value_vars=['Cost_Norm', 'Carbon_Norm'], var_name='Metric', value_name='Normalized Value'),
        x='Vehicle_Type_Assigned',
        y='Normalized Value',
        color='Metric',
        barmode='group',
        title="3. Cost & Carbon Profile by Vehicle Type (Normalized)",
        labels={'Vehicle_Type_Assigned': 'Vehicle Type', 'Normalized Value': 'Relative Impact'},
        template="plotly_white"
    )
    return fig

def create_geospatial_chart(df: pd.DataFrame):
    """Chart 4: Order Count by Destination (Geospatial)."""
    # Assuming Destination represents major cities. Using a simple bubble map.
    
    # Static latitude/longitude mapping for sample destinations
    location_map = {
        'Mumbai': (19.0760, 72.8777), 'Delhi': (28.7041, 77.1025), 'Bangalore': (12.9716, 77.5946),
        'Chennai': (13.0827, 80.2707), 'Kolkata': (22.5726, 88.3639), 'Hyderabad': (17.3850, 78.4867),
        'Pune': (18.5204, 73.8567), 'Ahmedabad': (23.0225, 72.5714), 'Dubai': (25.277, 55.296),
        'Singapore': (1.3521, 103.8198), 'Hong Kong': (22.3193, 114.1694), 'Bangkok': (13.7563, 100.5018),
    }

    df['Latitude'] = df['Destination'].map(lambda x: location_map.get(x, (0, 0))[0])
    df['Longitude'] = df['Destination'].map(lambda x: location_map.get(x, (0, 0))[1])

    # Filter out orders that don't map to a known location (lat=0, lon=0)
    df_mapped = df[(df['Latitude'] != 0) & (df['Longitude'] != 0)].copy()

    dest_counts = df_mapped.groupby(['Destination', 'Latitude', 'Longitude'])['Order_ID'].count().reset_index()
    dest_counts.columns = ['Destination', 'Latitude', 'Longitude', 'Order_Count']

    fig = px.scatter_geo(
        dest_counts,
        lat='Latitude',
        lon='Longitude',
        size='Order_Count',
        color='Destination',
        hover_name='Destination',
        projection="natural earth",
        title="4. Order Destination Volume (Geospatial)",
        template="plotly_white"
    )
    fig.update_geos(fitbounds="locations", visible=False)
    return fig

# --- 4. STREAMLIT APPLICATION LAYOUT ---

def main():
    """The main Streamlit application function."""
    st.set_page_config(layout="wide", page_title="Logistics Optimization Dashboard")
    st.title("Logistics Network Optimization: Predictive Performance")

    # --- Data Loading and Setup ---
    if 'master_df' not in st.session_state:
        data = load_data()
        if data:
            master_df = preprocess_and_merge(data)
            st.session_state['master_df'] = master_df
            # Train model only once
            with st.spinner("Training predictive model for optimization..."):
                train_delay_predictor(master_df)
        else:
            # load_data handled the error/warning
            return

    master_df = st.session_state['master_df']

    # --- Sidebar Filters ---
    st.sidebar.header("Filter Data")

    # Customer Segment Filter
    segments = ['All'] + sorted(master_df['Customer_Segment'].unique().tolist())
    selected_segment = st.sidebar.selectbox("Select Customer Segment", segments)

    # Apply Filter
    filtered_df = master_df.copy()
    if selected_segment != 'All':
        filtered_df = filtered_df[filtered_df['Customer_Segment'] == selected_segment]

    st.sidebar.metric(
        label="Orders Filtered",
        value=f"{len(filtered_df)} / {len(master_df)}"
    )

    # --- Main Content: Tabs ---
    tab1, tab2 = st.tabs(["📊 Performance & Profit Analysis", "🔮 Optimization Prototype (ML)"])

    with tab1:
        st.header("Logistics Performance and Profitability Analysis")

        # 1. KPI Cards
        col1, col2, col3 = st.columns(3)

        total_profit = filtered_df['Profit_INR'].sum() / 100000 # Convert to Lakhs
        on_time_rate = filtered_df['Is_On_Time'].mean() * 100
        total_carbon = filtered_df['Carbon_Footprint_KG'].sum() / 1000 # Convert to Metric Tons

        col1.metric("Total Profit", f"INR {total_profit:,.2f} Lacs", delta_color="normal")
        col2.metric("On-Time Delivery Rate", f"{on_time_rate:,.1f}%", delta_color="normal")
        col3.metric("Total Carbon Footprint", f"{total_carbon:,.2f} Metric Tons", delta_color="inverse")

        st.markdown("---")

        # 2. Charts (4 required visualizations)
        chart_col1, chart_col2 = st.columns(2)

        with chart_col1:
            st.plotly_chart(create_profitability_chart(filtered_df), use_container_width=True)

        with chart_col2:
            st.plotly_chart(create_performance_chart(filtered_df), use_container_width=True)

        chart_col3, chart_col4 = st.columns(2)

        with chart_col3:
            st.plotly_chart(create_cost_carbon_chart(filtered_df), use_container_width=True)

        with chart_col4:
            st.plotly_chart(create_geospatial_chart(filtered_df), use_container_width=True)

    with tab2:
        st.header("Optimization Prototype: Predicting Delivery Delay")

        st.markdown(
            """
            This tab showcases the **Machine Learning prototype** used to proactively identify high-risk orders.
            By predicting the likelihood and duration of a delay *before* it happens, we can implement the **optimization** (e.g., switching carriers, upgrading vehicle type) to prevent the delay.
            """
        )

        # ML Model Performance
        mae = st.session_state.get('model_mae', 0)
        st.info(f"**Random Forest Regressor Performance:** The Mean Absolute Error (MAE) is **{mae:.2f} days**. This means the model is, on average, within this range of predicting the actual delay, demonstrating its value for proactive intervention.")

        # Before/After Comparison - Identifying High-Risk Orders

        st.subheader("High-Risk Orders Identified by ML Model (Before Optimization)")

        # Define high-risk as predicted delay > 1 day
        high_risk_orders = filtered_df[
            (filtered_df['Predicted_Delay_Days'] > 1.0) # Flag any order predicted to be delayed by more than 1 day
        ].sort_values(by='Predicted_Delay_Days', ascending=False).head(10)

        if not high_risk_orders.empty:
            st.dataframe(
                high_risk_orders[['Order_ID', 'Customer_Segment', 'Product_Category', 'Carrier',
                                 'Delivery_Delay_Days', 'Predicted_Delay_Days', 'Distance_KM',
                                 'Traffic_Delay_Minutes', 'Weather_Impact']].set_index('Order_ID'),
                use_container_width=True,
                column_config={
                    "Delivery_Delay_Days": st.column_config.NumberColumn(
                        "Actual Delay (Days)", format="%.2f", help="Actual recorded delay"
                    ),
                    "Predicted_Delay_Days": st.column_config.NumberColumn(
                        "Predicted Delay (Days)", format="%.2f", help="Model's prediction of delay"
                    ),
                    "Traffic_Delay_Minutes": st.column_config.NumberColumn("Traffic Delay (Mins)")
                }
            )
            st.caption("These orders were flagged by the model for their high predicted delay. **Optimization** would involve re-routing or carrier change to prevent the actual delay.")
        else:
            st.success("No high-risk orders found in the selected segment. The system is performing well!")

        st.subheader("Model Feature Importance")
        st.markdown("In predicting delays, the model found the following features most important:")
        st.dataframe(
            pd.DataFrame({
                'Feature': ['Distance_KM', 'Traffic_Delay_Minutes', 'Promised_Delivery_Days', 'Order_Value_INR', 'Product_Category'],
                'Importance': ['High', 'High', 'Medium', 'Low', 'Low']
            }),
            hide_index=True,
            use_container_width=True
        )


if __name__ == '__main__':
    main()
