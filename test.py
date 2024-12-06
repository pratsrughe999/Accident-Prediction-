import pandas as pd

# Load the dataset
data = pd.read_csv('Cleaned_Accident.csv')

# Display column names and sample data
print(data.columns)
print(data.head())
