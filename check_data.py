"""
Quick data inspection for the Superstore CSV.
Run: python check_data.py
"""

import pandas as pd

df = pd.read_csv("data/Sample - Superstore.csv", encoding="latin1")
df.columns = df.columns.str.strip()
df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")

print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nMissing values:")
print(df.isnull().sum())
print("\nDate range:", df["Order Date"].min(), "to", df["Order Date"].max())
print("Customers:", df["Customer ID"].nunique())
print("Orders:", df["Order ID"].nunique())
print("Products:", df["Product ID"].nunique())
print("Categories:", sorted(df["Category"].dropna().unique().tolist()))
print("Discount min/max:", df["Discount"].min(), df["Discount"].max())
print("\nFirst rows:")
print(df.head())
