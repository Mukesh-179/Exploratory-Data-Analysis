import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
df = pd.read_csv("D:\Java\cleaned_dataset.csv")   # Use cleaned dataset from Task 1
print("\n===== Dataset Loaded =====")
print(df.head())
print("\n===== Summary Statistics =====")
print(df.describe())
print("\n===== Column-wise Mean =====")
print(df.mean())
print("\n===== Column-wise Median =====")
print(df.median())
print("\n===== Column-wise Standard Deviation =====")
print(df.std())
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
df[numeric_cols].hist(figsize=(12, 10), bins=20)
plt.suptitle("Histograms of Numerical Features")
plt.show()
plt.figure(figsize=(12, 6))
sns.boxplot(data=df[numeric_cols])
plt.title("Boxplots for Numerical Features")
plt.show()
sns.pairplot(df[numeric_cols], diag_kind='hist')
plt.show()
corr = df.corr()
plt.figure(figsize=(10, 7))
sns.heatmap(corr, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()
col1 = numeric_cols[0]
col2 = numeric_cols[1]
fig = px.scatter(df, x=col1, y=col2, title=f"{col1} vs {col2} Scatter Plot")
fig.show()
fig = px.histogram(df, x=col1, title=f"Distribution of {col1}")
fig.show()
print("\n===== Observations =====")
print("- Check histograms to understand feature distributions.")
print("- Boxplots reveal outliers and spread of values.")
print("- Pairplots show relationships and clustering patterns.")
print("- Correlation heatmap identifies strongly related features.")
print("- Use scatterplots to confirm linear/non-linear trends.")
print("\n🎉 Task 2 EDA Completed Successfully!")
