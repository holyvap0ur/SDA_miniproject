# =====================================================================
# Advanced End-to-End Data Preprocessing: Autonomous Radar Telemetry
# =====================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 1. Load Dataset
print("Loading dataset...")
df = pd.read_csv('autonomous_radar_telemetry_50k.csv')
print(f"Original Dataset Shape: {df.shape}")

# Create working copy
df_clean = df.copy()

# =====================================================================
# STEP A: Deep Structural Cleaning & Duplicate Checks
# =====================================================================
print("\n--- Step A: Structural Cleaning ---")
missing_count = df_clean.isnull().sum().sum()
duplicate_count = df_clean.duplicated().sum()
print(f"Total Missing Values: {missing_count}")
print(f"Total Duplicate Records: {duplicate_count}")

if duplicate_count > 0:
    df_clean.drop_duplicates(inplace=True)
    print("Duplicates removed.")


# =====================================================================
# STEP B: Comprehensive Outlier Detection and Treatment (IQR Capping)
# =====================================================================
print("\n--- Step B: Outlier Treatment (Winsorization) Across All Parameters ---")

# Select all numerical parameters except identifiers/flags
num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
excluded_cols = ['sensor_degraded_flag']
target_num_cols = [col for col in num_cols if col not in excluded_cols]

outlier_summary = {}
for col in target_num_cols:
    Q1 = df_clean[col].quantile(0.25)
    Q3 = df_clean[col].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Count outliers before treatment
    outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
    outlier_summary[col] = len(outliers)
    
    # Treat outliers via Capping (Winsorization)
    df_clean[col] = df_clean[col].clip(lower_bound, upper_bound)

print("Outlier counts detected per feature:")
for feature, count in outlier_summary.items():
    if count > 0:
        print(f"  - {feature}: {count} outliers capped")


# =====================================================================
# STEP C: Data Transformation & Distribution Normalization
# =====================================================================
print("\n--- Step C: Transformation of Skewed Features ---")

# Apply log transformation (log1p) to highly right-skewed variables
skewed_features = ['precipitation_rate_mmh', 'packet_latency_ms', 'ambient_lux']
for col in skewed_features:
    if col in df_clean.columns:
        df_clean[f'log_{col}'] = np.log1p(df_clean[col])
        print(f"Log transformation applied to: {col}")


# =====================================================================
# STEP D: Normalization and Standardization (Scaling)
# =====================================================================
print("\n--- Step D: Min-Max Normalization & Z-Score Standardization ---")

# Key telemetry parameters for scaling
scale_features = [
    'vehicle_speed_kmh', 'tx_power_dbm', 'snr_db', 
    'target_range_meters', 'sensor_temp_c', 'mcu_voltage_v'
]

scaler_minmax = MinMaxScaler()
scaler_std = StandardScaler()

# Generate normalized [0, 1] and standardized (mean=0, std=1) columns
df_clean[[f'{col}_norm' for col in scale_features]] = scaler_minmax.fit_transform(df_clean[scale_features])
df_clean[[f'{col}_std' for col in scale_features]] = scaler_std.fit_transform(df_clean[scale_features])
print("Scaling completed for core numerical features.")


# =====================================================================
# STEP E: Domain-Specific Feature Engineering
# =====================================================================
print("\n--- Step E: Advanced Feature Engineering ---")

# 1. Transmitter Power to MCU Voltage Efficiency Ratio
df_clean['power_to_voltage_ratio'] = df_clean['tx_power_dbm'] / df_clean['mcu_voltage_v']

# 2. Thermal Health Status Categorization
df_clean['thermal_status'] = pd.cut(
    df_clean['sensor_temp_c'],
    bins=[-np.inf, 40, 60, np.inf],
    labels=['Normal', 'High', 'Critical']
)

# 3. Signal-to-Noise Ratio (SNR) Quality Tier
df_clean['snr_category'] = pd.cut(
    df_clean['snr_db'],
    bins=[-np.inf, 10, 20, np.inf],
    labels=['Low', 'Medium', 'High']
)

# 4. Range-Speed Interaction Index
df_clean['range_speed_interaction'] = df_clean['target_range_meters'] * df_clean['vehicle_speed_kmh']

print("New engineered features added: power_to_voltage_ratio, thermal_status, snr_category, range_speed_interaction.")


# =====================================================================
# STEP F: Final Verification & Export
# =====================================================================
print("\n--- Step F: Final Dataset Summary ---")
print(f"Final Preprocessed Dataset Shape: {df_clean.shape}")
print(f"Missing Values Check: {df_clean.isnull().sum().sum()}")
print(f"Duplicate Records Check: {df_clean.duplicated().sum()}")

# Export cleaned dataset
output_file = 'fully_cleaned_autonomous_radar_telemetry.csv'
df_clean.to_csv(output_file, index=False)
print(f"\nSuccess! Fully preprocessed and cleaned dataset exported to '{output_file}'.")