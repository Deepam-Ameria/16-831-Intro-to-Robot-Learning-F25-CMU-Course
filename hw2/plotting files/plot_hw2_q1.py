import os
import pandas as pd
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
LOG_DIR = 'data'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)

def load_tfevents_data(log_dir_path):
    """
    Reads all scalar events from a TensorBoard log directory.
    Returns a DataFrame containing the 'Eval_AverageReturn' vs. 'Iteration'.
    """
    event_file = None
    # Find the .tfevents file inside the directory
    for f in os.listdir(log_dir_path):
        if f.startswith('events.out.tfevents'):
            event_file = os.path.join(log_dir_path, f)
            break
    
    if not event_file:
        print(f"No .tfevents file found in {log_dir_path}. Skipping.")
        return None

    # Use EventAccumulator to load the TensorBoard events
    event_acc = EventAccumulator(event_file)
    event_acc.Reload()

    # The tag we are interested in is 'Eval_AverageReturn'
    tag = 'Eval_AverageReturn'
    if tag not in event_acc.Tags()['scalars']:
        print(f"Tag '{tag}' not found in {event_file}. Skipping.")
        return None
    
    # Extract data
    scalar_events = event_acc.Scalars(tag)
    
    # Create DataFrame: step is the iteration number, value is the return
    data = {
        'Iteration': [event.step for event in scalar_events],
        'Eval_AverageReturn': [event.value for event in scalar_events]
    }
    
    return pd.DataFrame(data)

def get_experiment_details(filename):
    """Extracts experiment details (rtg, dsa/na, batch size) from the filename."""
    parts = filename.split('_')
    batch_type = parts[1] # 'lb' or 'sb'
    
    if 'no_rtg' in filename and 'dsa' in filename:
        label = 'No RTG + DSA (Full Trajectory + Baseline)'
    elif 'rtg_dsa' in filename:
        label = 'RTG + DSA (Reward-to-Go + Baseline)'
    elif 'rtg_na' in filename:
        label = 'RTG + No Baseline (Reward-to-Go Only)'
    else:
        label = 'Unknown Configuration'
        
    # Standardize 'lb' and 'sb' labels
    batch_size_map = {'lb': 'Large Batch (6000)', 'sb': 'Small Batch (1500)'}
    return batch_type, batch_size_map.get(batch_type, 'Unknown'), label
# Define a standard smoothing window size
SMOOTHING_WINDOW = 10 

def plot_returns(batch_type, experiment_data):
    """
    Generates the plot for the specified batch type, 
    applying a rolling mean for smoothing.
    """
    plt.figure(figsize=(10, 6))
    
    # Plotting loop: data is stored as [(df, batch_size_str, label), ...]
    for df, batch_size_str, label in experiment_data[batch_type]:
        
        # 1. APPLY ROLLING MEAN FOR SMOOTHING
        # Calculate the rolling mean of the evaluation return
        smoothed_return = df['Eval_AverageReturn'].rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1, # Start smoothing from the first data point
            center=True    # Center the window for better visualization
        ).mean()

        # Plot the smoothed data
        plt.plot(df['Iteration'], smoothed_return, label=label)

    batch_size_str = experiment_data[batch_type][0][1] # Get descriptive name
    
    plt.xlabel('Iteration')
    plt.ylabel(f'Evaluation Average Return (Smoothed by {SMOOTHING_WINDOW} Iterations)')
    plt.title(f'Policy Gradient Performance ({batch_size_str}) - CartPole-v0')
    plt.legend()
    plt.grid(True, alpha=0.5)
    
    # Save the plot
    output_filename = f'pg_returns_{batch_type}_smoothed.png' # Changed filename to reflect smoothing
    plt.savefig(os.path.join(PLOTS_DIR, output_filename))
    plt.close()
    print(f"Plot saved: {os.path.join(PLOTS_DIR, output_filename)}")

# --- MAIN EXECUTION (Updated to include SMOOTHING_WINDOW) ---
if __name__ == '__main__':
    
    # Define a standard smoothing window size before main execution
    SMOOTHING_WINDOW = 10 
    
    data_for_plotting = {'lb': [], 'sb': []}
    
    # ... (rest of the data loading remains the same)
    log_directories = [d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]
    
    for dirname in log_directories:
        full_path = os.path.join(LOG_DIR, dirname)
        batch_type, batch_size_str, label = get_experiment_details(dirname)
        
        df = load_tfevents_data(full_path)
        
        if df is not None:
            data_for_plotting[batch_type].append((df, batch_size_str, label))

    if data_for_plotting['sb']:
        plot_returns('sb', data_for_plotting)
        
    if data_for_plotting['lb']:
        plot_returns('lb', data_for_plotting)

    print("TFEVENTS plotting complete.")