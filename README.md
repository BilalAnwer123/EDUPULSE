<<<<<<< HEAD
# Student Risk System

Early-warning tool that shows student performance patterns and predicts whether a student is at risk of failing.

## Use it (no installation)
Open `app/index.html` in any browser (double-click it).
- **Dashboard** tab: pass/fail counts, fail rate by attendance and quiz average, pass vs. fail profile, model accuracy/precision/recall and confusion matrix.
- **Predict risk** tab: enter one student's values, get pass/fail, the probability, the top factors and plain-language advice.

The charts load a small library from the internet the first time, so connect to the internet when you open it.

## Update it with new data
1. Install Python packages once: `pip install pandas scikit-learn`
2. Replace the CSV in `data/` (same column names), or point to a new file.
3. Run from this folder: `python src/build_app.py` (or `python src/build_app.py path/to/new_data.csv`)
4. Reopen `app/index.html`. The cleaned data, model and dashboard are all refreshed.

## Folder guide
| Path | What it is |
|---|---|
| `app/index.html` | The finished app |
| `data/` | Original CSV and the cleaned CSV |
| `src/build_app.py` | Cleans data, trains the model, builds the app |
| `src/app_template.html` | App design (edit this to change how the app looks) |
| `experiments/` | Extra model scripts (Logistic Regression, Extra Trees, Random Forest) |

## Notes
- Required columns: student_id, attendance_pct, quiz_avg, assignment_avg, mid_marks, lab_marks, study_hours_week, prev_cgpa, part_time_job (Yes/No), final_result (Pass/Fail).
- A student is flagged as at risk when the predicted probability of failing is 50% or more.
- The prediction is an estimate from past data. Use it together with a teacher's or advisor's judgement.
=======
# EDUPULSE
>>>>>>> 802be48805c88b102e4f0b97dd7e85549cb013db
