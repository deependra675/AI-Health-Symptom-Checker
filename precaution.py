import pandas as pd
import joblib

# Loading dataset
df = pd.read_csv(r"C:\Users\SiS\Desktop\np\dataset\symptom_precaution.csv")

# Clean column names
df.columns = df.columns.str.strip()

# Checking required columns
if "Disease" not in df.columns:
    raise ValueError("CSV must have Disease column")

# Fill missing values
df = df.fillna("")

# create dict.
precaution_dict = {}

for _, row in df.iterrows():
    disease = row["Disease"]
    precautions = []

    # collect all precaution columns dynamically
    for col in df.columns:
        if "Precaution" in col:
            value = row[col]
            if value and str(value).strip() != "":
                precautions.append(value.strip())
    precaution_dict[disease] = precautions

# Saving processed file
joblib.dump(precaution_dict, "precaution.pkl")
print("saved successfully")