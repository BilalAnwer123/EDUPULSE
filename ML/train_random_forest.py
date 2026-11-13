"""Train a Random Forest to predict student Fail (the at-risk class).
Works with either the raw CSV or the cleaned CSV. Imputation happens INSIDE the
pipeline, so medians come from the training data only (no test-set leakage)."""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

DATA = "Track_4_-_student_performance.csv"   # change to your file path
SEED = 42

df = pd.read_csv(DATA).drop_duplicates()
df = df.drop(columns=[c for c in df.columns if c.endswith("_was_missing")])
y = (df["final_result"] == "Fail").astype(int)           # 1 = Fail (at risk)
X = df.drop(columns=["student_id", "final_result"])
X["part_time_job"] = (X["part_time_job"] == "Yes").astype(int)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=SEED)   # same 80/20 split as the other script

model = make_pipeline(
    SimpleImputer(strategy="median"),
    RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=SEED),
)
model.fit(X_train, y_train)
pred = model.predict(X_test)

print(f"Train: {len(X_train)} rows | Test: {len(X_test)} rows | Fail in test: {y_test.sum()}\n")
cm = confusion_matrix(y_test, pred)
print("== Random Forest ==")
print(pd.DataFrame(cm, index=["Actual Pass", "Actual Fail"], columns=["Pred Pass", "Pred Fail"]), "\n")

res = pd.DataFrame([{"Model": "Random Forest",
                     "Accuracy": accuracy_score(y_test, pred),
                     "Precision": precision_score(y_test, pred, zero_division=0),
                     "Recall": recall_score(y_test, pred)}]).round(3)
print(res.to_string(index=False))
res.to_csv("random_forest_results.csv", index=False)

# Which features matter most to the model
rf = model.named_steps["randomforestclassifier"]
imp = pd.Series(rf.feature_importances_, index=X.columns).sort_values(ascending=False).round(3)
print("\nFeature importance:\n" + imp.to_string())
