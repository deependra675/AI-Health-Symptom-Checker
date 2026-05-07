import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib

# Load training data
df = pd.read_csv(r"C:\Users\SiS\Desktop\np\dataset\dataset.csv")

# clean columns if spaces
df.columns = df.columns.str.strip()

# getting all unique symptoms
all_symptoms = set()
symptom_columns = df.columns[1:] # skip Disease column

for col in symptom_columns:
    df[col] = df[col].astype(str).str.strip()
    all_symptoms.update(df[col].dropna().unique())

# Removing invalid values
all_symptoms.discard("nan")
all_symptoms.discard("None")
all_symptoms = sorted(all_symptoms)

# Creating binary matrix
encoded_rows = []
for _, row in df.iterrows():
    symptom_dict = {symptom: 0 for symptom in all_symptoms}
    for col in symptom_columns:
        symptom = str(row[col]).strip()

        if symptom in symptom_dict:
            symptom_dict[symptom] = 1

    symptom_dict["Disease"] = row['Disease']
    encoded_rows.append(symptom_dict)

# final dataframe
encoded_df = pd.DataFrame(encoded_rows)

# Features and target
X = encoded_df.drop("Disease", axis=1)
y = encoded_df["Disease"]

# train_test_split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

# Train model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)
model.fit(X_train, y_train)

# Accuracy
accuracy = model.score(X_test, y_test)
print("Accuracy:", accuracy)

# model saving
joblib.dump(model, "model.pkl")
joblib.dump(X.columns.tolist(), "symptoms.pkl")

