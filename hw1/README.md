## Resources for learning about PyTorch and Gym
- For pytorch:  
	- https://www.cs.utexas.edu/~yukez/cs391r_fall2021/slides/tutorial_09-29_pytorch_intro.pdf#page=6.00  
	- https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html  
	- https://rail.eecs.berkeley.edu/deeprlcourse/deeprlcourse/static/slides/lec-3.pdf#page=25.00  
	- https://cs230.stanford.edu/blog/pytorch/  
- For openai gym:  
	- https://www.gocoder.one/blog/rl-tutorial-with-openai-gym/  
	- https://www.gymlibrary.dev/content/tutorials/  
	- https://blog.paperspace.com/getting-started-with-openai-gym/  


## Setup

You can run this code locally on your own machine or on Google Colab. We recommend Python 3.8+ on a linux machine.  

1. **Local option:** If you choose to run locally, you will need to install MuJoCo and some Python packages; see [installation.md](installation.md) for instructions.
2. **Colab:** The first few sections of the notebook will install all required dependencies and pull the repo. You can try out the Colab option by clicking the badge below:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/LeCAR-Lab/16831-F25-HW/blob/main/hw1/rob831/scripts/run_hw1.ipynb)

Note: If running in colab and you pull the repo again or install new packages, you may need to restart the runtime and reconnect before being able to run.

## Complete the code

Detailed instructions about the assignment and what you need to do are in the homework pdf.

Fill in sections of the code marked with `TODO`. In particular, see
 - [infrastructure/rl_trainer.py](rob831/infrastructure/rl_trainer.py)
 - [policies/MLP_policy.py](rob831/policies/MLP_policy.py)
 - [infrastructure/replay_buffer.py](rob831/infrastructure/replay_buffer.py)
 - [infrastructure/utils.py](rob831/infrastructure/utils.py)
 - [infrastructure/pytorch_util.py](rob831/infrastructure/pytorch_util.py)

Look for sections maked with `HW1` to see how the edits you make will be used.
Some other files that you may find relevant
 - [scripts/run_hw1.py](rob831/scripts/run_hw1.py) (if running locally) or [scripts/run_hw1.ipynb](rob831/scripts/run_hw1.ipynb) (if running on Colab)
 - [agents/bc_agent.py](rob831/agents/bc_agent.py)


## Run the code

Tip: While debugging, you probably want to keep the flag `--video_log_freq -1` which will disable video logging and speed up the experiment. However, feel free to remove it to save videos of your awesome policy!

If running on Colab, adjust the `#@params` in the `Args` class according to the commmand line arguments above.

If you do not have GPU available, you can run the code by adding the `--no_gpu` flag. It will still run but will be slower.

### Section 1 (Behavior Cloning)
General command for section 1:

```
python rob831/scripts/run_hw1.py \
	--expert_policy_file rob831/policies/experts/Ant.pkl \
	--env_name Ant-v2 --exp_name bc_ant --n_iter 1 \
	--expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
	--learning_rate 4e-3 --n_layers 5 --video_log_freq -1 
```

Make sure to follow the homework PDF for more details on what else you need to run.
To generate videos of the policy, remove the `--video_log_freq -1` flag.

### Section 2 (DAgger)
General command for section 2:
(Note the `--do_dagger` flag, and the higher value for `n_iter`)

```
python rob831/scripts/run_hw1.py \
    --expert_policy_file rob831/policies/experts/Ant.pkl \
    --env_name Ant-v2 --exp_name dagger_ant --n_iter 8 \
    --do_dagger --expert_data rob831/expert_data/expert_data_Ant-v2.pkl \
	--n_layers 3 --video_log_freq -1
```

Make sure to also try another environment.
See the homework PDF for more details on what else you need to run.

## Visualizing with TensorBoard:

You can visualize your runs using tensorboard by running:
```
python -m tensorboard.main --logdir data
```

You will see scalar summaries, graphs, as well as videos of your trained policies (in the 'images' tab) of the dashboard.

You can choose to visualize specific runs with a comma-separated list:
```
python -m tensorboard.main --logdir data/run1,data/run2,data/run3...
```

## Viewing Logged Videos
Logging videos is optional but can help with visualizing your results and debugging! To log videos, run the code commands above without the `--video_log_freq -1` flag.

The videos will be logged to the tfevents file in your `hw1/data/[run]/` folder and can be viewed in the TensorBoard dashboard. Keep in mind that videos will only be logged if you're running more than one training iteration.

If you encounter a `moviepy` error, try reinstalling the package:
```
pip uninstall moviepy
pip install moviepy==1.0.3
```