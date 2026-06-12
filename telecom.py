import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv(
    r"C:\Users\hp\OneDrive\New folder\OneDrive\Desktop\Telecom+Customer+Churn\telecom_customer_cleaned_churn.csv"
)


# Dataset Overview

print(df.head())

print("\nShape:")
print(df.shape)

print("\nColumns:")
print(df.columns)

print("\nInfo:")
print(df.info())

print("\nSummary Statistics:")
print(df.describe())

# Missing Values

print("\nMissing Values:")
print(df.isnull().sum())

# Duplicate Records

print("\nDuplicate Rows:")
print(df.duplicated().sum())

# Customer Status

print("\nCustomer Status:")
print(df["Customer Status"].value_counts())

print("\nCustomer Status %:")
print(
    round(
        df["Customer Status"].value_counts(normalize=True) * 100,
        2
    )
)

sns.countplot(data=df, x="Customer Status")
plt.title("Customer Status Distribution")
plt.show()

# Churn Rate

churn_rate = (
    df[df["Customer Status"] == "Churned"].shape[0]
    / df.shape[0]
) * 100

print("\nChurn Rate:")
print(round(churn_rate, 2))

# Revenue Analysis

print("\nTotal Revenue:")
print(df["Total Revenue"].sum())

revenue_status = (
    df.groupby("Customer Status")
    ["Total Revenue"]
    .sum()
)

print("\nRevenue by Customer Status:")
print(revenue_status)

revenue_status.plot(kind="bar")
plt.title("Revenue by Customer Status")
plt.ylabel("Revenue")
plt.show()

# Churn Data

churn = df[
    df["Customer Status"] == "Churned"
]

print("\nTotal Churned Customers:")
print(churn.shape[0])

# Contract Analysis

print("\nContract Analysis:")
print(churn["Contract"].value_counts())

sns.countplot(
    data=churn,
    x="Contract"
)

plt.title("Contract Analysis")
plt.show()

# Internet Type Analysis

print("\nInternet Type Analysis:")
print(churn["Internet Type"].value_counts())

sns.countplot(
    data=churn,
    x="Internet Type"
)

plt.title("Internet Type Analysis")
plt.show()

# Churn Categories

print("\nChurn Categories:")
print(churn["Churn Category"].value_counts())

plt.figure(figsize=(8,5))

sns.countplot(
    data=churn,
    y="Churn Category",
    order=churn["Churn Category"]
    .value_counts()
    .index
)

plt.title("Churn Categories")
plt.show()

# Top Churn Reasons

print("\nTop Churn Reasons:")
print(
    churn["Churn Reason"]
    .value_counts()
    .head(10)
)

plt.figure(figsize=(10,5))

sns.countplot(
    data=churn,
    y="Churn Reason",
    order=churn["Churn Reason"]
    .value_counts()
    .head(10)
    .index
)

plt.title("Top 10 Churn Reasons")
plt.show()

# Gender Analysis

print("\nGender Distribution:")
print(df["Gender"].value_counts())

df["Gender"].value_counts().plot(
    kind="pie",
    autopct="%1.1f%%"
)

plt.title("Gender Distribution")
plt.ylabel("")
plt.show()

# Age Analysis

print("\nAge Statistics:")
print(df["Age"].describe())

sns.histplot(
    df["Age"],
    bins=10
)

plt.title("Age Distribution")
plt.show()

# Revenue by Contract

contract_revenue = (
    df.groupby("Contract")
    ["Total Revenue"]
    .sum()
)

print("\nRevenue by Contract:")
print(contract_revenue)

contract_revenue.plot(kind="bar")

plt.title("Revenue by Contract")
plt.ylabel("Revenue")
plt.show()

# Revenue by Internet Type

internet_revenue = (
    df.groupby("Internet Type")
    ["Total Revenue"]
    .sum()
)

print("\nRevenue by Internet Type:")
print(internet_revenue)

internet_revenue.plot(kind="bar")

plt.title("Revenue by Internet Type")
plt.ylabel("Revenue")
plt.show()

# Top 10 Cities by Revenue

city_revenue = (
    df.groupby("City")
    ["Total Revenue"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

print("\nTop 10 Cities by Revenue:")
print(city_revenue)

city_revenue.plot(kind="barh")

plt.title("Top 10 Cities by Revenue")
plt.xlabel("Revenue")
plt.show()

# Monthly Charge Analysis

monthly_charge = (
    df.groupby("Customer Status")
    ["Monthly Charge"]
    .mean()
)

print("\nAverage Monthly Charge:")
print(monthly_charge)

# Findings

print("\nKEY FINDINGS")

print("1. Churn rate is around 26%.")
print("2. Month-to-Month customers show higher churn.")
print("3. Fiber Optic customers contribute significant churn.")
print("4. Competitor-related reasons are major churn drivers.")
print("5. Long-term contracts have better retention.")
print("6. Revenue loss due to churn is substantial.")
print("7. Top cities contribute a major share of revenue.")