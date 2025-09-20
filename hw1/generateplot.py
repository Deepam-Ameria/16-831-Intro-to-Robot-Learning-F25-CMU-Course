import matplotlib.pyplot as plt
import numpy as np

# Data extracted from the provided terminal output for Hopper-v2
iterations = np.arange(10)
dagger_avg_returns = np.array([844.5543, 1077.3502, 2653.9475, 3160.6912, 3779.6458, 3768.4546, 3775.3914, 3779.4683, 3771.7554, 3779.9883])
dagger_std_returns = np.array([319.5888, 138.3159, 600.2012, 516.2390, 6.9010, 6.6145, 6.0175, 4.5754, 3.0764, 2.0727])

# Expert and Initial BC returns as horizontal lines
expert_return = 3772.6704
initial_bc_return = 844.5543

# Create the plot
plt.figure(figsize=(10, 6))

# Plot the DAgger learning curve with error bars
plt.errorbar(iterations, dagger_avg_returns, yerr=dagger_std_returns, fmt='o-', capsize=5, label='DAgger Policy Mean Return')

# Plot horizontal lines for the expert and initial BC policies
plt.axhline(y=expert_return, color='r', linestyle='--', label='Expert Policy')
plt.axhline(y=initial_bc_return, color='g', linestyle='--', label='Initial BC Policy')

# Set plot title and labels
plt.title('DAgger Learning Curve for Hopper-v2 Environment')
plt.xlabel('DAgger Iterations')
plt.ylabel('Policy Mean Return')
plt.legend()
plt.grid(True)
plt.savefig('dagger_hopper_learning_curve.png')