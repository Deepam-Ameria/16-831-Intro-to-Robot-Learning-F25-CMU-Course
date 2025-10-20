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

# Directories you want (just those 4)
TARGET_DIRS = [
    "q4_b25000_r0.02_HalfCheetah-v4_02-10-2025_18-14-28",
    "q4_b25000_r0.02_nnbaseline_HalfCheetah-v4_02-10-2025_18-15-41",
    "q4_b25000_r0.02_rtg_HalfCheetah-v4_02-10-2025_18-14-44",
    "q4_b25000_r0.02_rtg_nnbaseline_HalfCheetah-v4_02-10-2025_18-16-43"
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
    plt.plot(df['Iteration'], smoothed, label=dirname)

plt.xlabel("Iteration")
plt.ylabel(f"Eval_AverageReturn (smoothed {SMOOTHING_WINDOW})")
plt.title(f"Q4 Comparison - {ENV_NAME}")
plt.legend(fontsize="small")
plt.grid(True, alpha=0.5)

out_path = os.path.join(PLOTS_DIR, "q4_four_runs.png")
plt.savefig(out_path)
plt.close()
print(f"Plot saved: {out_path}")
