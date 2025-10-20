import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
LOG_DIR = 'data'   # Put your q2 folders inside here
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
SMOOTHING_WINDOW = 10   # smaller window for faster envs
ENV_NAME = "InvertedPendulum-v4"
TARGET_PREFIX = 'q2_'   # only look at q2 runs


def load_tfevents_data(log_dir_path):
    """Load Eval_AverageReturn data from TensorBoard tfevents file."""
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
        print(f"Error loading {log_dir_path}: {e}")
        return None

    tag = 'Eval_AverageReturn'
    if tag not in event_acc.Tags()['scalars']:
        return None

    scalar_events = event_acc.Scalars(tag)

    df = pd.DataFrame({
        'Iteration': [event.step for event in scalar_events],
        'Value': [event.value for event in scalar_events]
    })

    clean_df = df.groupby('Iteration')['Value'].mean().reset_index()
    clean_df.rename(columns={'Value': 'Eval_AverageReturn'}, inplace=True)

    return clean_df


def parse_q2_details(dirname):
    """
    Parse batch size and learning rate from q2 folder name.
    Example: q2_b10000_r0.01_InvertedPendulum-v4_01-10-2025_21-12-20
    """
    if not dirname.startswith(TARGET_PREFIX):
        return None

    parts = dirname.split('_')
    try:
        batch_size = parts[1].replace('b', '')   # b10000 → 10000
        learning_rate = parts[2].replace('r', '')  # r0.01 → 0.01
        label = f"B:{batch_size}, LR:{learning_rate}"
        return label
    except IndexError:
        print(f"Could not parse details from {dirname}")
        return None


def plot_q2_all_runs(experiment_data_list):
    """
    Plot all Q2 runs on one figure for comparison.
    """
    plt.figure(figsize=(10, 6))

    for df, label in experiment_data_list:
        smoothed_return = df['Eval_AverageReturn'].rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=True
        ).mean()
        plt.plot(df['Iteration'], smoothed_return, label=label)

    plt.xlabel("Iteration")
    plt.ylabel(f"Eval_AverageReturn (Smoothed {SMOOTHING_WINDOW})")
    plt.title(f"Q2: Hyperparameter Comparison - {ENV_NAME}")
    plt.legend(title="Parameters", fontsize="small")
    plt.grid(True, alpha=0.5)

    output_file = os.path.join(PLOTS_DIR, "q2_all_runs_comparison.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Plot saved: {output_file}")


def plot_single_best(df, best_label):
    """
    Plot just the best run separately.
    """
    plt.figure(figsize=(10, 6))
    smoothed_return = df['Eval_AverageReturn'].rolling(
        window=SMOOTHING_WINDOW,
        min_periods=1,
        center=True
    ).mean()
    plt.plot(df['Iteration'], smoothed_return, color="blue", label=best_label)

    plt.xlabel("Iteration")
    plt.ylabel(f"Eval_AverageReturn (Smoothed {SMOOTHING_WINDOW})")
    plt.title(f"Q2: Best Run - {ENV_NAME}")
    plt.legend()
    plt.grid(True, alpha=0.5)

    output_file = os.path.join(PLOTS_DIR, f"q2_best_run_{best_label}.png")
    plt.savefig(output_file)
    plt.close()
    print(f"Best run plot saved: {output_file}")


# --- MAIN ---
if __name__ == "__main__":
    all_runs = []

    for dirname in os.listdir(LOG_DIR):
        full_path = os.path.join(LOG_DIR, dirname)
        if not os.path.isdir(full_path):
            continue

        label = parse_q2_details(dirname)
        if not label:
            continue

        df = load_tfevents_data(full_path)
        if df is not None:
            all_runs.append((df, label))

    # Plot all runs together
    if all_runs:
        plot_q2_all_runs(all_runs)

        # --- Pick best run (by final average return) ---
        best_df, best_label = max(all_runs, key=lambda x: x[0]['Eval_AverageReturn'].iloc[-1])
        plot_single_best(best_df, best_label)

    print("\n--- Q2 plotting complete ---")
