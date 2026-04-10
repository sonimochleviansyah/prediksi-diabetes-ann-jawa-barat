import pandas as pd

df = pd.read_csv("diabetes_jabar.csv")

print("Nama kolom:")
print(df.columns)

print("\n5 data pertama:")
print(df.head())

print("\nInfo dataset:")
print(df.info())

print("\nMissing value:")
print(df.isnull().sum())
