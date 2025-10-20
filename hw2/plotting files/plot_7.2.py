import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIG ---
LOG_DIR = 'data'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
SMOOTHING_WINDOW = 25
ENV_NAME = "HalfCheetah-v4"

# Target directories (Q4 search runs)
TARGET_DIRS = [
    "q4_search_b10000_lr0.02_HalfCheetah-v4_02-10-2025_20-19-14",
    "q4_search_b10000_lr0.02_nnbaseline_HalfCheetah-v4_02-10-2025_21-02-42",
    "q4_search_b10000_lr0.02_rtg_HalfCheetah-v4_02-10-2025_20-20-27",
    "q4_search_b10000_lr0.02_rtg_nnbaseline_HalfCheetah-v4_02-10-2025_21-07-17"
]

def load_tfevents_data(log_dir_path):
    event_file = None
    for f in os.listdir(log_dir_path):
        if f.startswith('events.out.tfevents'):
            event_file = os.path.join(log_dir_path, f)
            break
    if not event_file:
        return None

    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    tag = 'Eval_AverageReturn'
    if tag not in event_acc.Tags()['scalars']:
        return None

    scalar_events = event_acc.Scalars(tag)
    df = pd.DataFrame({
        'Iteration': [event.step for event in scalar_events],
        'Value': [event.value for event in scalar_events]
    })
    return df.groupby('Iteration')['Value'].mean().reset_index()

# --- MAIN ---
plt.figure(figsize=(10, 6))

for dirname in TARGET_DIRS:
    full_path = os.path.join(LOG_DIR, dirname)
    df = load_tfevents_data(full_path)
    if df is None:
        continue

    smoothed = df['Value'].rolling(
        window=SMOOTHING_WINDOW, min_periods=1, center=True
    ).mean()

    # Make labels short (baseline, nnbaseline, rtg, rtg+nnbaseline)
    if "nnbaseline" in dirname and "rtg" in dirname:
        label = "rtg+nnbaseline"
    elif "nnbaseline" in dirname:
        label = "nnbaseline"
    elif "rtg" in dirname:
        label = "rtg"
    else:
        label = "baseline"

    plt.plot(df['Iteration'], smoothed, label=label)

plt.xlabel("Iteration")
plt.ylabel(f"Eval_AverageReturn (smoothed {SMOOTHING_WINDOW})")
plt.title(f"Q4: Comparison (B=10000, LR=0.02) - {ENV_NAME}")
plt.legend(title="Variant", fontsize="small")
plt.grid(True, alpha=0.5)

out_path = os.path.join(PLOTS_DIR, "q4_b10000_lr0.02_comparison.png")
plt.savefig(out_path)
plt.close()
print(f"Plot saved: {out_path}")
