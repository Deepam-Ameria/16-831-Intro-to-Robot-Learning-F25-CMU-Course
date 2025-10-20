import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
LOG_DIR = 'data'   # base folder where your q3 run lives
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
SMOOTHING_WINDOW = 25  # smoothing for noisy environments
ENV_NAME = "LunarLanderContinuous-v2"
TARGET_PREFIX = 'q3_'   # single-run prefix

def load_tfevents_data(log_dir_path):
    """
    Reads the 'Eval_AverageReturn' scalar events from a TensorBoard log directory.
    """
    event_file = None
    for f in os.listdir(log_dir_path):
        if f.startswith('events.out.tfevents'):
            event_file = os.path.join(log_dir_path, f)
            break
    
    if not event_file:
        return None

    try:
        event_acc = EventAccumulator(event_file)
        event_acc.Reload()
    except Exception as e:
        print(f"Error loading EventAccumulator for {log_dir_path}: {e}")
        return None

    tag = 'Eval_AverageReturn'
    
    if tag not in event_acc.Tags()['scalars']:
        return None
    
    scalar_events = event_acc.Scalars(tag)

    # Convert events to DataFrame
    full_df = pd.DataFrame({
        'Iteration': [event.step for event in scalar_events],
        'Value': [event.value for event in scalar_events]
    })

    # Group by iteration to handle duplicates
    clean_df = full_df.groupby('Iteration')['Value'].mean().reset_index()
    clean_df.rename(columns={'Value': 'Eval_AverageReturn'}, inplace=True)
    
    return clean_df


def plot_q3_single_run(df, run_name):
    """
    Generates a single plot for a single Q3 run.
    """
    plt.figure(figsize=(10, 6))

    # Apply rolling mean smoothing
    smoothed_return = df['Eval_AverageReturn'].rolling(
        window=SMOOTHING_WINDOW,
        min_periods=1,
        center=True
    ).mean()

    plt.plot(df['Iteration'], smoothed_return, label=run_name, color="blue")

    plt.xlabel('Iteration')
    plt.ylabel(f'Evaluation Average Return (Smoothed by {SMOOTHING_WINDOW})')
    plt.title(f'Q3: {ENV_NAME} - {run_name}')
    plt.legend()
    plt.grid(True, alpha=0.5)

    # Save plot
    output_filename = f'{run_name}_smoothed.png'
    plt.savefig(os.path.join(PLOTS_DIR, output_filename))
    plt.close()
    print(f"Plot saved: {os.path.join(PLOTS_DIR, output_filename)}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':

    # 1. Gather experiment directories inside LOG_DIR
    log_directories = [d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]

    for dirname in log_directories:
        if not dirname.startswith(TARGET_PREFIX):
            continue
        
        run_name = dirname
        full_path = os.path.join(LOG_DIR, dirname)

        # 2. Load and parse TFEVENTS data
        df = load_tfevents_data(full_path)

        if df is not None:
            # 3. Plot the single run
            plot_q3_single_run(df, run_name)

    print("\n--- Q3 plotting complete ---")
