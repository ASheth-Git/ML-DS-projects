#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 30 19:19:44 2025

@author: alpesh
"""
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the CSV file
Wafer_df = pd.read_csv('Wafer_Dataset.csv')



# =============================================================================
# Step 1
# =============================================================================
# =============================================================================
# Determine which lot have more defect
# =============================================================================

# Group by 'lot_id' and sum 'defect_count'
lot_defects = Wafer_df.groupby('lot_id')['defect_count'].sum().reset_index()

# Sort the result in descending order
lot_defects_sorted = lot_defects.sort_values(by='defect_count', ascending=False)

# Display the top lots with most defects
print(lot_defects_sorted.head())

# Optional: Plot the top 10 defective lots
plt.figure(figsize=(12, 6))
sns.barplot(data=lot_defects_sorted.head(10), x='lot_id', y='defect_count', palette='Reds_d')
plt.title('Top Lots with Most Defects')
plt.xlabel('Lot ID')
plt.ylabel('Total Defect Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Group by 'process_stage' and sum 'defect_count'
stage_defects = Wafer_df.groupby('process_stage')['defect_count'].sum().reset_index()

# Sort by defect count descending for better visualization
stage_defects_sorted = stage_defects.sort_values(by='defect_count', ascending=False)

# Display the summarized data
print(stage_defects_sorted)

# Plot total defects by process stage
plt.figure(figsize=(10, 6))
sns.barplot(data=stage_defects_sorted, x='process_stage', y='defect_count', palette='Blues_d')
plt.title('Total Defect Count by Process Stage')
plt.xlabel('Process Stage')
plt.ylabel('Total Defect Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
    

# Standardize the process_stage names
Wafer_df['process_stage'] = Wafer_df['process_stage'].replace({'Etch': 'Etching'})

# Now group by 'process_stage' and sum 'defect_count'
stage_defects = Wafer_df.groupby('process_stage')['defect_count'].sum().reset_index()

# Sort by defect count descending for better visualization
stage_defects_sorted = stage_defects.sort_values(by='defect_count', ascending=False)

# Display the summarized data
print(stage_defects_sorted)

# Plot total defects by process stage
plt.figure(figsize=(10, 6))
sns.barplot(data=stage_defects_sorted, x='process_stage', y='defect_count', palette='Blues_d')
plt.title('Total Defect Count by Process Stage')
plt.xlabel('Process Stage')
plt.ylabel('Total Defect Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Step 2: Keep only relevant stages
target_stages = ['Etching', 'Deposition', 'Lithography', 'Implantation']
filtered_df = Wafer_df[Wafer_df['process_stage'].isin(target_stages)]

# Step 3: Group by lot_id and process_stage, sum defect_count
grouped = filtered_df.groupby(['lot_id', 'process_stage'])['defect_count'].sum().reset_index()

# Step 4: Pivot to get process_stages as columns
pivot_df = grouped.pivot(index='lot_id', columns='process_stage', values='defect_count').fillna(0).astype(int)

# Display the result
print(pivot_df.head())

# Step 4: Bar plot (grouped bar chart)
plt.figure(figsize=(14, 8))
sns.barplot(data=grouped, x='lot_id', y='defect_count', hue='process_stage')
plt.title('Defect Count per Process Stage by Lot')
plt.xlabel('Lot ID')
plt.ylabel('Defect Count')
plt.xticks(rotation=45)
plt.legend(title='Process Stage')
plt.tight_layout()
plt.show()

for column in Wafer_df.columns:
    num_missing = Wafer_df[column].isna().sum()
    num_empty_strings = (Wafer_df[column] == '').sum() if Wafer_df[column].dtype == 'object' else 0
    total_empty = num_missing + num_empty_strings
    print(f"{column}: {total_empty} empty entries")

# Replace missing values in 'pressure_mbar' with the column mean
mean_pressure = Wafer_df['pressure_mbar'].astype(float).mean()
Wafer_df['pressure_mbar'] = Wafer_df['pressure_mbar'].fillna(mean_pressure)

# Replace missing and empty string entries in 'defect_type' with 'NoDefects'

# First, handle missing or empty values only in defect_type
Wafer_df['defect_type'] = Wafer_df['defect_type'].replace('', pd.NA)

# Now fill missing values conditionally based on defect_count
Wafer_df['defect_type'] = Wafer_df.apply(
    lambda row: 'NoDefects' if pd.isna(row['defect_type']) and row['defect_count'] == 0
    else ('UndeterminedDefects' if pd.isna(row['defect_type']) else row['defect_type']),
    axis=1
)


for column in Wafer_df.columns:
    num_missing = Wafer_df[column].isna().sum()
    num_empty_strings = (Wafer_df[column] == '').sum() if Wafer_df[column].dtype == 'object' else 0
    total_empty = num_missing + num_empty_strings
    print(f"New  {column}: {total_empty} empty entries")


# Group by defect_type and sum defect_count
defect_type_grouped = Wafer_df.groupby('defect_type')['defect_count'].sum().reset_index()

# Sort for cleaner visualization
defect_type_grouped = defect_type_grouped.sort_values(by='defect_count', ascending=False)

# Plot
plt.figure(figsize=(14, 6))
sns.barplot(data=defect_type_grouped, x='defect_type', y='defect_count', palette='viridis')
plt.title('Total Defect Count per Defect Type')
plt.xlabel('Defect Type')
plt.ylabel('Total Defect Count')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# Group and summarize
grouped_df = Wafer_df.groupby(['lot_id', 'process_stage', 'defect_type'])['defect_count'].sum().reset_index()

# Create FacetGrid with hue (only here!)
g = sns.FacetGrid(grouped_df, col='process_stage', col_wrap=2, height=5, sharex=False, hue='defect_type', palette='tab10')

# Map the barplot WITHOUT hue here
g.map_dataframe(sns.barplot, x='lot_id', y='defect_count', dodge=True)

# Customize
g.set_titles(col_template='Process Stage: {col_name}')
g.set_axis_labels('Lot ID', 'Defect Count')
g.set_xticklabels(rotation=45)
g.add_legend(title='Defect Type')

plt.tight_layout()
plt.show()

# Group by defect_types and sum the defect_count
defect_summary = Wafer_df.groupby('defect_type')['defect_count'].sum().reset_index()

# Sort by defect_count for better readability
defect_summary = defect_summary.sort_values(by='defect_count', ascending=False)

# Display the result
print(defect_summary)

# Extract the substring before the first colon
Wafer_df['equipment_id'] = Wafer_df['equipment_log'].str.split(':').str[0]

# Optional: preview unique extracted equipment IDs
print(Wafer_df['equipment_id'].unique())

# Group and count number of entries per (equipment_id, process_stage)
equipment_stage_counts = Wafer_df.groupby(['equipment_id', 'process_stage']).size().reset_index(name='count')

# Plot
plt.figure(figsize=(14, 8))
sns.barplot(data=equipment_stage_counts, x='equipment_id', y='count', hue='process_stage')
plt.title('Process Stage Distribution per Equipment ID')
plt.xlabel('Equipment ID')
plt.ylabel('Number of Records')
plt.xticks(rotation=45)
plt.legend(title='Process Stage')
plt.tight_layout()
plt.show()

    

# # Optional: Check if there are still any missing values
# print(Wafer_df_cleaned.isnull().sum())

# unique_process_stages = Wafer_df_cleaned['process_stage'].unique()
# # print(unique_process_stages)
# stage_counts = Wafer_df_cleaned['process_stage'].value_counts()
# print(stage_counts)




# =============================================================================
# First way #Gives Warning 
# =============================================================================
# =============================================================================
# Replacing Mistake Etch with Etching
# =============================================================================

# Replace 'Etch' with 'Etching'
# Wafer_df_cleaned['process_stage'] = Wafer_df_cleaned['process_stage'].replace('Etch', 'Etching')

# Verify the fix
# print(Wafer_df_cleaned['process_stage'].unique())

# print(Wafer_df_cleaned['process_stage'].value_counts())
# =============================================================================
# End
# =============================================================================
# =============================================================================
# Other way
# =============================================================================
# Wafer_df_cleaned.loc[Wafer_df_cleaned['process_stage'] == 'Etch', 'process_stage'] = 'Etching'

# # Clean the DataFrame properly
# Wafer_df_cleaned = Wafer_df.dropna().copy()

# # Safely replace 'Etch' with 'Etching'
# Wafer_df_cleaned['process_stage'] = Wafer_df_cleaned['process_stage'].replace('Etch', 'Etching')

# print(Wafer_df_cleaned['process_stage'].unique())

# print(Wafer_df_cleaned['process_stage'].value_counts())
# =============================================================================
# End
# =============================================================================

# =============================================================================
# Checking other data and its uniqueness
# =============================================================================

# unique_defect_type=Wafer_df_cleaned['defect_type'].unique()
# print(unique_defect_type)
# stage_counts = Wafer_df_cleaned['defect_type'].value_counts()
# print(stage_counts)

# unique_wafer_id=Wafer_df_cleaned['wafer_id'].unique()
# print(unique_wafer_id)
# stage_counts = Wafer_df_cleaned['wafer_id'].value_counts()
# print(stage_counts)

# unique_equip_log=Wafer_df_cleaned['equipment_log'].unique()
# print(unique_equip_log)
# stage_counts = Wafer_df_cleaned['equipment_log'].value_counts()
# print(stage_counts)


# =============================================================================
# End 
# =============================================================================

# =============================================================================
# Checking correlation to determine dependency
# =============================================================================
# Compute correlation matrix
correlation_matrix = Wafer_df.corr(numeric_only=True)

# Plot heatmap to determine the correlation between data
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Wafer Dataset')
plt.show()

# Create the ratio column
Wafer_df['temperature_C_*_hour'] =Wafer_df['hour']*Wafer_df['temperature_C'] 

# Drop the individual columns
Wafer_df = Wafer_df.drop(columns=['temperature_C', 'hour'])

# Compute correlation matrix
correlation_matrix = Wafer_df.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Wafer Dataset (temperature_C/hour only)')
plt.show()


# Create the new feature column: pressure_mbar * exposure_time_ms
Wafer_df['pressure_*_exposure'] = Wafer_df['pressure_mbar']* Wafer_df['exposure_time_ms']

# Drop the individual columns
Wafer_df = Wafer_df.drop(columns=['pressure_mbar', 'exposure_time_ms'])

# Compute correlation matrix
correlation_matrix = Wafer_df.corr(numeric_only=True)

# Plot the heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Heatmap of Wafer Dataset (with pressure_*_exposure)')
plt.show()


# =============================================================================
# 
# =============================================================================


# import numpy as np

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# # Define features and target
# features = ['temperature_C', 'pressure_mbar', 'hour', 'exposure_time_ms']
# X = Wafer_df[features]
# y = Wafer_df['defect_count']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train model
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)

# # Predict
# y_pred = lin_reg.predict(X_test)

# # Evaluate
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Linear Regression Results:")
# print("MSE (lower is better) :", mse)
# print("R² score (near to 1 is better):", r2)

# import numpy as np
# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# # Create new feature
# Wafer_df['temperature_C_*_hour'] =Wafer_df['hour']*Wafer_df['temperature_C'] 

# # Drop individual columns (optional but recommended)
# Wafer_df = Wafer_df.drop(columns=['temperature_C', 'hour'])

# # Define new feature set and target
# features = ['temperature_C_*_hour', 'pressure_mbar', 'exposure_time_ms']
# X = Wafer_df[features]
# y = Wafer_df['defect_count']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train model
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)

# # Predict
# y_pred = lin_reg.predict(X_test)

# # Evaluate
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Linear Regression Results (with temperature_C_*_hour):")
# print("MSE (lower is better):", mse)
# print("R² score (near to 1 is better):", r2)

# =============================================================================
#  From heat map plot determined target that can be
#  predicted which is  defect_count based on 
#  temp , pressure , hour , exposure time
# =============================================================================
# import numpy as np

# from sklearn.linear_model import LinearRegression
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error, r2_score

# # Define features and target
# features = ['temperature_C', 'pressure_mbar', 'hour', 'exposure_time_ms']
# X = Wafer_df[features]
# y = Wafer_df['defect_count']

# # Train-test split
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Initialize and train model
# lin_reg = LinearRegression()
# lin_reg.fit(X_train, y_train)

# # Predict
# y_pred = lin_reg.predict(X_test)

# # Evaluate
# mse = mean_squared_error(y_test, y_pred)
# r2 = r2_score(y_test, y_pred)

# print("Linear Regression Results:")
# print("MSE (lower is better) :", mse)
# print("R² score (near to 1 is better):", r2)



# =============================================================================
# From above MSE and R^2 determined Linear Regression is 
# Not so good model because R^{2}  is of order ~ 0.1
# Better if R^{2} is near 1
# =============================================================================

# =============================================================================
# Checking skewness , checking  features importance
# =============================================================================

# sns.histplot(y, bins=30, kde=True)
# plt.show()

# importances = model.feature_importances_
# plt.barh(features, importances)
# plt.title("Feature Importance")
# plt.show()

# =============================================================================
# 
# =============================================================================

# =============================================================================
# From above it is determined that selected Feautures
# are decent enough and carry equal weight and also 
# Not many skewed value i.e. 0 in target   
# =============================================================================
