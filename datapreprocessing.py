import pandas as pd

# Read the CSV file and convert it into a pandas DataFrame
file_path = "data/games.csv"
df = pd.read_csv(file_path)

# Display the first few rows of the DataFrame
print(df["moves"].head())