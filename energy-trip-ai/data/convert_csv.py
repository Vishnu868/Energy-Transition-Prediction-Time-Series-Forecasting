import pandas as pd

input_file = ""
output_file = "sample_ukdale.csv"

# Read file
df = pd.read_csv(input_file, sep=r"\s+", header=None)

# Assign column names
df.columns = ["timestamp", "active_power", "apparent_power", "voltage"]

# Keep only timestamp and active_power
df = df[["timestamp", "active_power"]]

# Rename to match expected format
df.columns = ["timestamp", "power"]

# Save
df.to_csv(output_file, index=False)

print("✅ Converted to expected format!")