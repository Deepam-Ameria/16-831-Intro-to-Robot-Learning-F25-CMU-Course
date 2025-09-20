## Running instructions for HW1

### Q1 - Behavior Cloning

I used the following commands to run Behaviour Cloning on Ant-v2 and Hopper-v2 environments (with tuned params):

#### Ant

 $ python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 1 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --learning_rate 5e-3 --n_layers 4 --video_log_freq -1 --eval_batch_size 5000 --train_batch_size 300

#### Hopper 

 $ python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 1 --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --learning_rate 5e-3 --n_layers 4 --video_log_freq -1 --eval_batch_size 5000 --train_batch_size 300

These commands will generate results that are tabulated in Part 3 (Comparison of Ant and Another Env).

To re-generate results of Part 4, I ran the same Ant-v2 command with different learning rates and plotted the result.

### Q2 - DAgger

I used the following commands to run DAgger on both Ant-v2 and Hopper-v2 environments (with 10 iterations and eval_batch_size = 5000):

#### Ant

 $ python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Ant.pkl --env_name Ant-v2 --exp_name bc_ant --n_iter 10 --expert_data rob831/expert_data/expert_data_Ant-v2.pkl --do_dagger --video_log_freq -1 --eval_batch_size 5000

#### Hopper
 $ python rob831/scripts/run_hw1.py --expert_policy_file rob831/policies/experts/Hopper.pkl --env_name Hopper-v2 --exp_name bc_hopper --n_iter 10 --expert_data rob831/expert_data/expert_data_Hopper-v2.pkl --do_dagger --video_log_freq -1 --eval_batch_size 5000

These commands will generate results that were used to create the plots for Part 2 (Learning Curves using DAgger for Ant-v2 and Hopper v2 environments with standard deviation as error bars)