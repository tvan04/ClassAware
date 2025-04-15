import os
import csv
import random
import numpy as np
from datetime import datetime, timedelta

# ----------------------------
# CONFIGURATION
# ----------------------------
RANDOM_SEED = 12323
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

LOGS_FOLDER = "Student Logs"
os.makedirs(LOGS_FOLDER, exist_ok=True)

NUM_STUDENTS = 1000
NUM_TEACHER_EVENTS = 50
START_TIME = datetime(2025, 1, 1, 9, 0, 0)

LABEL_MAP = {"Low": 0, "Medium": 1, "High": 2}
INV_LABEL_MAP = {v: k for k, v in LABEL_MAP.items()}

EVENT_TYPES = [
    "No match detected",
    "No match detected (error)",
    "Raised Hand detected",
    "Thumbs Up detected",
    "Thumbs Down detected"
]

TEACHER_ACTIONS = ["Ask Question", "Ask Yes/No"]
TEACHER_LOG_FILE = "teacher_log.csv"

# ----------------------------
# Generate Teacher Log
# ----------------------------
def generate_teacher_log():
    current_time = START_TIME
    log = []
    for _ in range(NUM_TEACHER_EVENTS):
        offset = random.randint(3, 8)
        current_time += timedelta(minutes=offset)
        action = random.choice(TEACHER_ACTIONS)
        log.append((current_time.strftime("%Y-%m-%d %H:%M:%S"), action))

    with open(TEACHER_LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Timestamp", "Action"])
        writer.writerows(log)

    print(f"Generated teacher log: {TEACHER_LOG_FILE}")
    return log

# ----------------------------
# Simulate Student Logs
# ----------------------------
def simulate_student_logs(teacher_log, num_students=NUM_STUDENTS):
    label_choices = ["Low", "Medium", "High"]
    label_probs = [0.3, 0.4, 0.3]

    for sid in range(num_students):
        student_id = f"S{sid+1}"
        label = random.choices(label_choices, weights=label_probs, k=1)[0]
        label_num = LABEL_MAP[label]

        filepath = os.path.join(LOGS_FOLDER, f"{student_id}.csv")
        with open(filepath, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([f"ENGAGEMENT_LABEL: {label}"])
            writer.writerow(["Timestamp", "Event"])

            for timestamp_str, action in teacher_log:
                t = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S")
                respond = False
                delay_seconds = 0
                noise_count = int(np.random.poisson(1))

                # Human behavior simulation
                if label_num == 2:
                    respond = random.random() < 0.92 + random.uniform(-0.05, 0.03)
                    delay_seconds = random.choice([1, 2, 3])
                elif label_num == 1:
                    respond = random.random() < 0.75 + random.uniform(-0.1, 0.1)
                    delay_seconds = random.choice([2, 3, 5, 7])
                else:
                    respond = random.random() < 0.35 + random.uniform(-0.15, 0.1)
                    delay_seconds = random.choice([6, 8, 10, 15]) if respond else 0

                if respond:
                    event_time = t + timedelta(seconds=int(delay_seconds))
                    if action == "Ask Question":
                        writer.writerow([event_time.strftime("%Y-%m-%d %H:%M:%S"), "Raised Hand detected"])
                    elif action == "Ask Yes/No":
                        writer.writerow([event_time.strftime("%Y-%m-%d %H:%M:%S"),
                                         random.choice(["Thumbs Up detected", "Thumbs Down detected"])] )

                # Add realistic noise
                for _ in range(noise_count):
                    noise_offset = random.randint(10, 300)  # noise after 10 to 300 seconds
                    noise_event = random.choices(
                        ["No match detected", "No match detected (error)", "Thumbs Up detected"],
                        weights=[0.6, 0.2, 0.2],
                        k=1
                    )[0]
                    noise_time = t + timedelta(seconds=int(noise_offset))
                    writer.writerow([noise_time.strftime("%Y-%m-%d %H:%M:%S"), noise_event])

        print(f"[{student_id}] => label={label}, file={filepath}")

# ----------------------------
# Main
# ----------------------------
def main():
    print("Generating teacher log...")
    teacher_log = generate_teacher_log()

    print("\nGenerating student logs based on teacher prompts...")
    simulate_student_logs(teacher_log)

    print("\nDone. Check 'Student Logs/' and 'teacher_log.csv'.")

if __name__ == "__main__":
    main()
