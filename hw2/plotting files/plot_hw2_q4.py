import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# --- CONFIGURATION ---
LOG_DIR = 'data'
PLOTS_DIR = 'plots'
os.makedirs(PLOTS_DIR, exist_ok=True)
SMOOTHING_WINDOW = 25 # Increased smoothing for continuous control environments (HalfCheetah)
ENV_NAME = "HalfCheetah-v4"
TARGET_PREFIX = 'q4_search_' # Target prefix for Q4 files


def load_tfevents_data(log_dir_path):
    """
    Reads the 'Eval_AverageReturn' scalar events from a TensorBoard log directory.
    
    NOTE: Includes aggressive grouping to handle potential duplicate entries 
    within the TFEVENTS file structure.
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

    # Create DataFrame from all events
    full_df = pd.DataFrame({
        'Iteration': [event.step for event in scalar_events],
        'Value': [event.value for event in scalar_events]
    })

    # Group by iteration and take the mean to consolidate noisy/duplicate entries
    clean_df = full_df.groupby('Iteration')['Value'].mean().reset_index()
    clean_df.rename(columns={'Value': 'Eval_AverageReturn'}, inplace=True)
    
    return clean_df

def get_q4_experiment_details(filename):
    """
    Extracts batch size and learning rate for Q4 files.
    """
    
    if not filename.startswith(TARGET_PREFIX):
        return None

    # Expected format: q4_search_bXXXXX_rYYYY_...
    parts = filename.split('_')
    
    try:
        batch_size = parts[2].replace('b', '') # parts[2] = bXXXXX
        learning_rate = parts[3].replace('lr', '') # parts[3] = lrYYYY
        
        # The plot_group is now static for a single plot
        plot_group = "All Runs"
        
        # New comprehensive label showing both hyperparameters
        label = f'B:{batch_size}, LR:{learning_rate}'
        
    except IndexError:
        print(f"Could not parse Q4 details from: {filename}. Skipping.")
        return None

    return plot_group, label # Returns two values

def plot_q4_all_returns(experiment_data_list):
    """
    Generates a single plot for ALL Q4 runs, comparing different Batch and LR combinations.
    
    experiment_data_list is a list of tuples: [(df, plot_group_name, label), ...]
    """
    plt.figure(figsize=(12, 8)) # Use a larger figure for many lines
    
    # Plotting loop
    for df, _, label in experiment_data_list:
        
        # APPLY ROLLING MEAN FOR SMOOTHING
        smoothed_return = df['Eval_AverageReturn'].rolling(
            window=SMOOTHING_WINDOW,
            min_periods=1,
            center=True
        ).mean()

        # Plot the smoothed data, ensuring no markers are used
        plt.plot(df['Iteration'], smoothed_return, label=label, marker=None) 

    plt.xlabel('Iteration')
    plt.ylabel(f'Evaluation Average Return (Smoothed by {SMOOTHING_WINDOW} Iterations)')
    plt.title(f'Q4: Hyperparameter Search Comparison - {ENV_NAME}')
    plt.legend(title='Run Parameters', loc='upper left', fontsize='small') # Place legend where it won't obscure lines
    plt.grid(True, alpha=0.5)
    
    # Save the plot 
    output_filename = f'q4_all_runs_combined_smoothed.png' 
    plt.savefig(os.path.join(PLOTS_DIR, output_filename))
    plt.close()
    print(f"Plot saved: {os.path.join(PLOTS_DIR, output_filename)}")


# --- MAIN EXECUTION ---
if __name__ == '__main__':
    
    # List to store all Q4 runs together
    all_q4_runs = []
    
    # 1. Gather all experiment directories
    log_directories = [d for d in os.listdir(LOG_DIR) if os.path.isdir(os.path.join(LOG_DIR, d))]
    
    for dirname in log_directories:
        full_path = os.path.join(LOG_DIR, dirname)
        
        # 2. Check if it's a Q4 file and get details
        details = get_q4_experiment_details(dirname)
        
        if details is None:
            # Skip non-Q4 files
            continue
            
        plot_group, label = details 
        
        # 3. Load and parse the data from TFEVENTS
        df = load_tfevents_data(full_path)
        
        if df is not None:
            # Store data as (df, plot_group_name, label)
            all_q4_runs.append((df, plot_group, label))

    # 4. Generate the single combined plot
    if all_q4_runs:
        plot_q4_all_returns(all_q4_runs)

    print("\n--- Q4 Combined TFEVENTS plotting complete ---")
