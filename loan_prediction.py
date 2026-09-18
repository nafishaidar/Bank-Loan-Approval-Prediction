import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay

df = pd.read_csv("loan_prediction.csv")

for col in df.select_dtypes(include="object").columns:
    if col != "Loan_ID":
        df[col] = df[col].fillna(df[col].mode()[0])

for col in df.select_dtypes(include="number").columns:
    df[col] = df[col].fillna(df[col].median())

for col in ["Gender", "Married", "Education", "Self_Employed", "Property_Area", "Loan_Status"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])

X = df.drop(["Loan_ID", "Loan_Status"], axis=1)
y = df["Loan_Status"]

X = pd.get_dummies(X, columns=["Dependents"], dtype=int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print("Random Forest Loan Approval Prediction")
print("Accuracy:", round(accuracy * 100, 2), "%")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
plt.title("Loan Approval Confusion Matrix")
plt.show()

sample = X_test.iloc[[0]]
prediction = model.predict(sample)[0]

print("\nSample Prediction:", "Approved" if prediction == 1 else "Not Approved")
