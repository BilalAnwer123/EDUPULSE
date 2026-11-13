"""Train a Logistic Regression model to predict student Fail (the at-risk class).
Works with either the raw CSV or the cleaned CSV. Imputation happens INSIDE the
pipeline, so medians come from the training data only (no test-set leakage)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

DATA = "Track_4_-_student_performance.csv"   # change to your file path
SEED = 42

df = pd.read_csv(DATA).drop_duplicates()
df = df.drop(columns=[c for c in df.columns if c.endswith("_was_missing")])
y = (df["final_result"] == "Fail").astype(int)           # 1 = Fail (at risk)
X = df.drop(columns=["student_id", "final_result"])
X["part_time_job"] = (X["part_time_job"] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)   # 80/20, keeps Fail ratio equal

imp = lambda: SimpleImputer(strategy="median")
models = {
    "Logistic Regression": make_pipeline(imp(), StandardScaler(),
                                         LogisticRegression(class_weight="balanced", max_iter=1000)),
}

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Fail in test: {y_test.sum()}\n")
rows = []
for name, m in models.items():
    m.fit(X_train, y_train)
    pred = m.predict(X_test)
    cm = confusion_matrix(y_test, pred)
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": precision_score(y_test, pred, zero_division=0),
                 "Recall": recall_score(y_test, pred)})
    print(f"== {name} ==")
    print(pd.DataFrame(cm, index=["Actual Pass", "Actual Fail"], columns=["Pred Pass", "Pred Fail"]), "\n")

res = pd.DataFrame(rows).round(3)
print(res.to_string(index=False))
res.to_csv("model_results.csv", index=False)
