import os
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb

# -----------------------------------------
# CONFIG
# -----------------------------------------
LOGS_FOLDER = "Student Logs"
TEACHER_LOG_FILE = "teacher_log.csv"

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

# -----------------------------------------
# Load Teacher Log
# -----------------------------------------
def load_teacher_prompts():
    teacher_times = []
    if not os.path.exists(TEACHER_LOG_FILE):
        return teacher_times
    df = pd.read_csv(TEACHER_LOG_FILE)
    for _, row in df.iterrows():
        teacher_times.append({
            "timestamp": datetime.strptime(row["Timestamp"], "%Y-%m-%d %H:%M:%S"),
            "action": row["Action"]
        })
    return teacher_times

# -----------------------------------------
# Parse each student's CSV => response features
# -----------------------------------------
def load_labeled_data():
    teacher_prompts = load_teacher_prompts()
    if not teacher_prompts:
        print("No teacher log found. Exiting.")
        return pd.DataFrame()

    rows = []
    csv_files = glob(os.path.join(LOGS_FOLDER, "S*.csv"))
    csv_files.sort()

    for csv_path in csv_files:
        student_id = os.path.splitext(os.path.basename(csv_path))[0]
        with open(csv_path, "r", encoding="utf-8") as f:
            lines = f.read().splitlines()

        if not lines or not lines[0].startswith("ENGAGEMENT_LABEL:"):
            continue

        label_str = lines[0].replace("ENGAGEMENT_LABEL:", "").strip()
        if label_str not in LABEL_MAP:
            continue

        label = LABEL_MAP[label_str]
        data_lines = lines[2:]
        events = []
        no_match_count = 0

        for line in data_lines:
            parts = line.split(",", 1)
            if len(parts) < 2:
                continue
            try:
                timestamp = datetime.strptime(parts[0].strip(), "%Y-%m-%d %H:%M:%S")
                event = parts[1].strip()
                events.append((timestamp, event))
                if event == "No match detected":
                    no_match_count += 1
            except:
                continue

        response_data = {
            "StudentID": student_id,
            "on_time_responses": 0,
            "late_responses": 0,
            "missed_responses": 0,
            "no_match_count": no_match_count,
            "label": label
        }

        for prompt in teacher_prompts:
            prompt_time = prompt["timestamp"]
            expected_type = prompt["action"]
            if expected_type == "Ask Question":
                expected_events = ["Raised Hand detected"]
            elif expected_type == "Ask Yes/No":
                expected_events = ["Thumbs Up detected", "Thumbs Down detected"]
            else:
                continue

            matched = False
            for event_time, ev in events:
                if ev in expected_events:
                    delta = (event_time - prompt_time).total_seconds()
                    if 0 <= delta <= 3:
                        response_data["on_time_responses"] += 1
                        matched = True
                        break
                    elif 3 < delta <= 30:
                        response_data["late_responses"] += 1
                        matched = True
                        break
            if not matched:
                response_data["missed_responses"] += 1

        rows.append(response_data)

    return pd.DataFrame(rows)

# -----------------------------------------
# Train & Evaluate Model
# -----------------------------------------
def main():
    df = load_labeled_data()
    if df.empty:
        print("No valid student logs found.")
        return

    print("\nAggregated Response Data:\n", df.head())

    X = df[["on_time_responses", "late_responses", "missed_responses", "no_match_count"]]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

    rf_clf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_clf.fit(X_train, y_train)
    y_pred_rf = rf_clf.predict(X_test)

    acc_rf = accuracy_score(y_test, y_pred_rf)
    print("\n=== Random Forest Results ===")
    print("Accuracy:", acc_rf)
    print(classification_report(y_test, y_pred_rf, digits=4, target_names=["Low", "Medium", "High"]))

    xgb_clf = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss'
    )
    xgb_clf.fit(X_train, y_train)
    y_pred_xgb = xgb_clf.predict(X_test)

    acc_xgb = accuracy_score(y_test, y_pred_xgb)
    print("\n=== XGBoost Results ===")
    print("Accuracy:", acc_xgb)
    print(classification_report(y_test, y_pred_xgb, digits=4, target_names=["Low", "Medium", "High"]))

    print("\n=== Comparison Summary ===")
    print(f"Random Forest Accuracy: {acc_rf:.4f}")
    print(f"XGBoost Accuracy:      {acc_xgb:.4f}")

    df["RF_PredLabel"] = rf_clf.predict(X)
    df["RF_PredLabelStr"] = df["RF_PredLabel"].map(INV_LABEL_MAP)
    df["XGB_PredLabel"] = xgb_clf.predict(X)
    df["XGB_PredLabelStr"] = df["XGB_PredLabel"].map(INV_LABEL_MAP)

    output_csv = "final_labeled_results.csv"
    df.to_csv(output_csv, index=False)
    print(f"\nSaved predictions to '{output_csv}'")

if __name__ == "__main__":
    main()