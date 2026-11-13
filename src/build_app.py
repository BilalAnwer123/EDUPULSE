"""One-step pipeline: clean data -> train model -> build the app.

Usage (from the project folder):
    python src/build_app.py                      # uses data/Track_4_-_student_performance.csv
    python src/build_app.py path/to/new_data.csv # retrain on new data

Outputs:
    data/student_performance_cleaned.csv   cleaned dataset
    app/index.html                         the finished app (open in any browser)
"""
import sys, json, pathlib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

ROOT = pathlib.Path(__file__).resolve().parent.parent
DATA = pathlib.Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "data" / "Track_4_-_student_performance.csv"
SEED = 42
FEATURES = ["attendance_pct", "quiz_avg", "assignment_avg", "mid_marks", "lab_marks",
            "study_hours_week", "prev_cgpa", "part_time_job"]
MAXES = {"attendance_pct": 100, "quiz_avg": 10, "assignment_avg": 10, "mid_marks": 30, "lab_marks": 20, "prev_cgpa": 4}
LABELS = {"attendance_pct": "Attendance", "quiz_avg": "Quizzes", "assignment_avg": "Assignments",
          "mid_marks": "Mid marks", "lab_marks": "Lab marks", "prev_cgpa": "Prev. CGPA"}

# ---- 1. Clean ----------------------------------------------------------------
raw = pd.read_csv(DATA)
df = raw.drop_duplicates().reset_index(drop=True)
for c in ["part_time_job", "final_result"]:
    df[c] = df[c].str.strip().str.title()
num = [c for c in FEATURES if c != "part_time_job"]
cleaned = df.copy()
for c in num:                                    # flag, then fill with median
    cleaned[f"{c}_was_missing"] = cleaned[c].isna().astype(int)
cleaned[num] = cleaned[num].fillna(cleaned[num].median())
cleaned.to_csv(ROOT / "data" / "student_performance_cleaned.csv", index=False)
print(f"Cleaned: {len(raw)} -> {len(df)} rows ({len(raw)-len(df)} duplicates removed)")

# ---- 2. Train (Logistic Regression) -------------------------------------------
y = (df["final_result"] == "Fail").astype(int)   # 1 = Fail (at risk)
X = df[FEATURES].copy()
X["part_time_job"] = (X["part_time_job"] == "Yes").astype(int)
make = lambda: make_pipeline(SimpleImputer(strategy="median"), StandardScaler(),
                             LogisticRegression(class_weight="balanced", max_iter=1000))
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, stratify=y, random_state=SEED)
pred = make().fit(Xtr, ytr).predict(Xte)         # honest metrics on held-out data
metrics = {"accuracy": round(accuracy_score(yte, pred), 4), "precision": round(precision_score(yte, pred), 4),
           "recall": round(recall_score(yte, pred), 4), "cm": confusion_matrix(yte, pred).tolist(),
           "test_n": int(len(yte)), "test_fail": int(yte.sum())}
final = make().fit(X, y)                         # final model uses all rows
imp, sc, lr = (s for _, s in final.steps)
model = {"median": imp.statistics_.round(4).tolist(), "mean": sc.mean_.round(5).tolist(),
         "scale": sc.scale_.round(5).tolist(), "coef": lr.coef_[0].round(5).tolist(),
         "intercept": round(float(lr.intercept_[0]), 5)}
print("Held-out metrics:", {k: v for k, v in metrics.items() if k != "cm"})

# ---- 3. Dashboard numbers -----------------------------------------------------
fail = (cleaned["final_result"] == "Fail").astype(int)
def rate(col, bins, labels):
    g = fail.groupby(pd.cut(cleaned[col], bins=bins, labels=labels, right=False), observed=True).agg(["mean", "count"])
    return {"labels": labels, "vals": [round(v * 100, 1) for v in g["mean"]], "counts": g["count"].tolist()}
prof = (cleaned.groupby("final_result")[list(MAXES)].mean() / pd.Series(MAXES) * 100).round(1)
D = {"n": len(cleaned), "pass": int((fail == 0).sum()), "fail": int(fail.sum()),
     "avg_att": round(float(cleaned["attendance_pct"].mean()), 1),
     "att": rate("attendance_pct", [0, 60, 70, 80, 90, 101], ["<60", "60–69", "70–79", "80–89", "90+"]),
     "quiz": rate("quiz_avg", [0, 4, 5.5, 7, 8.5, 11], ["<4", "4–5.5", "5.5–7", "7–8.5", "8.5+"]),
     "profile": {"labels": [LABELS[c] for c in MAXES], "pass": prof.loc["Pass"].tolist(), "fail": prof.loc["Fail"].tolist()},
     "metrics": metrics, "model": model}

# ---- 4. Build the app ---------------------------------------------------------
html = (ROOT / "src" / "app_template.html").read_text(encoding="utf-8").replace("/*__DATA__*/null", json.dumps(D))
(ROOT / "app" / "index.html").write_text(html, encoding="utf-8")
print("Built app/index.html - open it in your browser.")
