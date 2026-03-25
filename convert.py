import pandas as pd
import json
import ast

# Read your corrected Excel file
df = pd.read_excel("disease_data.xlsx")

# Replace empty cells
df = df.fillna("")

# Columns that should stay as lists
list_columns = ["symptoms", "management"]

def convert_to_list(value):
    if isinstance(value, list):
        return value
    if isinstance(value, str) and value.strip():
        try:
            parsed = ast.literal_eval(value)
            if isinstance(parsed, list):
                return parsed
        except:
            pass
        return [value]
    return []

for col in list_columns:
    if col in df.columns:
        df[col] = df[col].apply(convert_to_list)

# Convert to list of dictionaries
data = df.to_dict(orient="records")

# Save as JSON
with open("disease_data.json", "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Excel converted to JSON successfully!")