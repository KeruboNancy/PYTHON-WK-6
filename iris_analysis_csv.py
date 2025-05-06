import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Note: Requires 'iris.csv' with columns: sepal_length, sepal_width, petal_length, petal_width, species
sns.set_style("whitegrid")

# Task 1: Load and Explore the Dataset
try:
    df = pd.read_csv('iris.csv')
    print("First 5 rows of the dataset:")
    print(df.head())
    print("\nDataset Info:")
    print(df.info())
    print("\nMissing Values:")
    print(df.isnull().sum())
    if df.isnull().sum().sum() > 0:
        print("\nHandling missing values by filling with column means...")
        df = df.fillna(df.select_dtypes(include=np.number).mean())
        print("Missing values after handling:")
        print(df.isnull().sum())
    else:
        print("\nNo missing values found.")
except FileNotFoundError:
    print("Error: 'iris.csv' file not found. Please ensure the file is in the same directory.")
    exit()
except pd.errors.EmptyDataError:
    print("Error: The CSV file is empty.")
    exit()
except Exception as e:
    print(f"Error loading dataset: {e}")
    exit()

# Task 2: Basic Data Analysis
try:
    print("\nBasic Statistics:")
    print(df.describe())
    print("\nMean values by species:")
    print(df.groupby('species').mean())
    print("\nObservations:")
    print("1. Setosa species tends to have smaller measurements overall.")
    print("2. Virginica generally has the largest petal measurements.")
    print("3. Versicolor falls between Setosa and Virginica in most measurements.")
except Exception as e:
    print(f"Error in data analysis: {e}")

# Task 3: Data Visualization
try:
    plt.figure(figsize=(15, 10))
    plt.subplot(2, 2, 1)
    means = df.groupby('species').mean()
    for column in means.columns:
        plt.plot(means.index, means[column], marker='o', label=column)
    plt.title('Mean Measurements by Species')
    plt.xlabel('Species')
    plt.ylabel('Measurement (cm)')
    plt.legend()
    plt.xticks(rotation=45)
    plt.subplot(2, 2, 2)
    sns.barplot(x='species', y='sepal_length', data=df)
    plt.title('Average Sepal Length by Species')
    plt.xlabel('Species')
    plt.ylabel('Sepal Length (cm)')
    plt.subplot(2, 2, 3)
    plt.hist(df['petal_length'], bins=20, color='skyblue')
    plt.title('Distribution of Petal Length')
    plt.xlabel('Petal Length (cm)')
    plt.ylabel('Frequency')
    plt.subplot(2, 2, 4)
    for species in df['species'].unique():
        species_data = df[df['species'] == species]
        plt.scatter(species_data['sepal_length'],
                    species_data['petal_length'],
                    label=species, alpha=0.6)
    plt.title('Sepal Length vs Petal Length')
    plt.xlabel('Sepal Length (cm)')
    plt.ylabel('Petal Length (cm)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('iris_visualizations.png')
    plt.close()
    print("\nVisualizations have been saved as 'iris_visualizations.png'")
except Exception as e:
    print(f"Error in visualization: {e}")

print("\nAdditional Findings:")
print("1. The scatter plot shows clear separation between species based on sepal and petal lengths.")
print("2. The histogram reveals a multimodal distribution of petal lengths, suggesting distinct species characteristics.")
print("3. The bar chart confirms Setosa has the shortest sepal length on average.")
print("4. The line chart shows consistent patterns across all measurements, with Virginica having the largest values.")
