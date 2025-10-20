import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIG ---
LOG_DIR = 'data'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
SMOOTHING_WINDOW = 25
ENV_NAME = "Hopper-v4"

# Directories you want (Q5 runs)
TARGET_DIRS = [
    "q5_b2000_r0.001_lambda0_Hopper-v4_02-10-2025_20-15-44",
    "q5_b2000_r0.001_lambda0.95_Hopper-v4_02-10-2025_20-15-52",
    "q5_b2000_r0.001_lambda0.99_Hopper-v4_02-10-2025_20-16-03",
    "q5_b2000_r0.001_lambda1_Hopper-v4_02-10-2025_20-16-18"
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

    # Shorter label (just lambda value)
    if "_lambda" in dirname:
        label = dirname.split("_lambda")[-1].split("_")[0]
        label = f"λ={label}"
    else:
        label = dirname

    plt.plot(df['Iteration'], smoothed, label=label)

plt.xlabel("Iteration")
plt.ylabel(f"Eval_AverageReturn (smoothed {SMOOTHING_WINDOW})")
plt.title(f"Q5: Effect of λ on {ENV_NAME}")
plt.legend(title="Lambda", fontsize="small")
plt.grid(True, alpha=0.5)

out_path = os.path.join(PLOTS_DIR, "q5_lambda_comparison.png")
plt.savefig(out_path)
plt.close()
print(f"Plot saved: {out_path}")
