import os
import numpy as np
import matplotlib.pyplot as plt
from tensorboard.backend.event_processing import event_accumulator

# Paths to your runs
base_dir = "data"
dqn_runs = [
    "q1_dqn_1_LunarLander-v3_30-10-2025_16-20-36",
    "q1_dqn_2_LunarLander-v3_30-10-2025_16-56-15",
    "q1_dqn_3_LunarLander-v3_30-10-2025_17-20-16",
]
ddqn_runs = [
    "q1_doubledqn_1_LunarLander-v3_30-10-2025_17-52-16",
    "q1_doubledqn_2_LunarLander-v3_30-10-2025_18-22-07",
    "q1_doubledqn_3_LunarLander-v3_30-10-2025_18-50-24",
]

def load_returns(run_dir, tag="Train_AverageReturn"):
    event_file = [
        f for f in os.listdir(run_dir) if f.startswith("events.out.tfevents")
    ][0]
    ea = event_accumulator.EventAccumulator(os.path.join(run_dir, event_file))
    ea.Reload()
    if tag not in ea.Tags()['scalars']:
        print(f"⚠️ Tag '{tag}' not found in {run_dir}. Available tags: {ea.Tags()['scalars']}")
        return None
    events = ea.Scalars(tag)
    steps = [e.step for e in events]
    values = [e.value for e in events]
    return np.array(steps), np.array(values)

def aggregate_runs(run_names, tag="Train_AverageReturn"):
    all_returns = []
    steps_ref = None
    for run in run_names:
        path = os.path.join(base_dir, run)
        data = load_returns(path, tag)
        if data is None:
            continue
        steps, returns = data
        if steps_ref is None:
            steps_ref = steps
        all_returns.append(np.interp(steps_ref, steps, returns))  # interpolate if needed
    all_returns = np.vstack(all_returns)
    mean = np.mean(all_returns, axis=0)
    std = np.std(all_returns, axis=0)
    return steps_ref, mean, std

# Extract data
steps_dqn, mean_dqn, std_dqn = aggregate_runs(dqn_runs)
steps_ddqn, mean_ddqn, std_ddqn = aggregate_runs(ddqn_runs)

# Plot
plt.figure(figsize=(8,5))
plt.plot(steps_dqn, mean_dqn, label="DQN", color="tab:blue")
plt.fill_between(steps_dqn, mean_dqn - std_dqn, mean_dqn + std_dqn, alpha=0.2, color="tab:blue")
plt.plot(steps_ddqn, mean_ddqn, label="Double DQN", color="tab:orange")
plt.fill_between(steps_ddqn, mean_ddqn - std_ddqn, mean_ddqn + std_ddqn, alpha=0.2, color="tab:orange")

plt.xlabel("Iterations")
plt.ylabel("Average Return")
plt.title("DQN vs Double DQN on LunarLander-v3")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
