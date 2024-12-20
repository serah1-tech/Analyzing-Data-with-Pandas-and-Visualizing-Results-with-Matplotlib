import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler

# Load the datasets
file1_path = 'downloads/dataset/181_plays_1585-610_t.csv'
file2_path = 'downloads/dataset/181_plays_1585-1610-Metadata.csv'

# Load datasets into DataFrames
df1 = pd.read_csv(file1_path)
df2 = pd.read_csv(file2_path)

# Task 1: Load and Explore the Dataset
# Display the first few rows of each dataset for initial inspection
df1_sample = df1.iloc[:, :10].head()  # Subset first 10 columns for readability
df2_sample = df2.head()

# Check for missing values in both datasets
df1_missing = df1.isnull().sum().sum()
df2_missing = df2.isnull().sum().sum()

# Drop rows/columns with too many missing values (threshold: 50%)
df1_cleaned = df1.dropna(axis=1, thresh=len(df1) * 0.5)
df2_cleaned = df2.dropna(axis=1, thresh=len(df2) * 0.5)

# Task 2: Basic Data Analysis
# Compute basic statistics for numerical columns
df1_stats = df1_cleaned.describe()

# Group by a categorical column from metadata and compute means (if applicable)
if 'Author' in df2_cleaned.columns:
    df2_grouped = df2_cleaned.groupby('Author').mean(numeric_only=True)
else:
    df2_grouped = None

# Task 3: Data Visualization
sns.set_theme(style="whitegrid")  # Set Seaborn theme for consistent styling

# 1. Line Chart (Trend over time)
if 'Date of first performance (best guess)' in df1_cleaned.columns:
    df1_sorted = df1_cleaned.sort_values('Date of first performance (best guess)')
    plt.figure(figsize=(10, 6))
    plt.plot(df1_sorted['Date of first performance (best guess)'].head(10),
             df1_sorted.iloc[:, 1].head(10),
             marker='o', label='Example Trend')
    plt.title('Trend Over Time')
    plt.xlabel('Year')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()

# 2. Bar Chart (Comparison of Numerical Values)
sample_columns = df1_cleaned.columns[1:6]
bar_data = df1_cleaned[sample_columns].mean()
plt.figure(figsize=(10, 6))
bar_data.plot(kind='bar', color='skyblue', title='Average Values of Selected Columns')
plt.xlabel('Words')
plt.ylabel('Average Frequency')
plt.show()

# 3. Histogram (Distribution of a Numerical Column)
plt.figure(figsize=(10, 6))
df1_cleaned.iloc[:, 1].plot(kind='hist', bins=15, color='purple', title='Distribution of Column 1')
plt.xlabel('Frequency')
plt.ylabel('Count')
plt.show()

# 4. Scatter Plot (Relationship between two columns)
plt.figure(figsize=(10, 6))
plt.scatter(df1_cleaned.iloc[:, 1], df1_cleaned.iloc[:, 2], alpha=0.5, color='green')
plt.title('Scatter Plot Between Column 1 and Column 2')
plt.xlabel('Column 1')
plt.ylabel('Column 2')
plt.show()

# Additional Analysis and Visualizations
# Correlation Heatmap
plt.figure(figsize=(12, 8))
correlation_matrix = df1_cleaned.corr()
sns.heatmap(correlation_matrix, annot=False, cmap='coolwarm')
plt.title('Correlation Heatmap')
plt.show()

# Pairplot for exploring relationships between columns
sns.pairplot(df1_cleaned.iloc[:, :6])  # Adjust to a subset of columns for clarity
plt.show()

# Normalized Data Distribution
scaler = MinMaxScaler()
normalized_data = pd.DataFrame(scaler.fit_transform(df1_cleaned.iloc[:, 1:6]), columns=df1_cleaned.columns[1:6])
plt.figure(figsize=(10, 6))
normalized_data.plot(kind='box', title='Normalized Data Distribution')
plt.show()

# Categorical Column Distribution (if applicable)
if 'Author' in df2_cleaned.columns:
    plt.figure(figsize=(10, 6))
    df2_cleaned['Author'].value_counts().plot(kind='bar', title='Distribution of Authors')
    plt.xlabel('Author')
    plt.ylabel('Count')
    plt.show()

# Results Summary
results = {
    'df1_sample': df1_sample,
    'df2_sample': df2_sample,
    'df1_missing': df1_missing,
    'df2_missing': df2_missing,
    'df1_stats': df1_stats,
    'df2_grouped': df2_grouped,
}

# Print summaries to validate
print("Dataset 1 Sample:")
print(df1_sample)
print("\nDataset 2 Sample:")
print(df2_sample)
print(f"\nMissing Values in Dataset 1: {df1_missing}")
print(f"Missing Values in Dataset 2: {df2_missing}")
if df2_grouped is not None:
    print("\nGrouped Data by Author:")
    print(df2_grouped)
