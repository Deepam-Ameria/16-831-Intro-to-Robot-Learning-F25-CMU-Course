## Install MuJoCo and other dependencies locally

There are two options:

A. (Recommended) Install with conda:

	1. Install conda, if you don't already have it, by following the instructions at [this link](https://docs.conda.io/projects/conda/en/latest/user-guide/install/)

	```

	This install will modify the `PATH` variable in your bashrc.
	You need to open a new terminal for that path change to take place (to be able to find 'conda' in the next step).

	2. Create a conda environment that will contain python 3:
	```
	conda create -n rob831 python=3.10
	```

	3. Activate the environment (do this every time you open a new terminal and want to run code):
	```
	source activate rob831
	```

	4. Install the requirements into this conda environment
	```
	pip install --user -r requirements.txt
	```

	5. Allow your code to be able to see 'rob831'
	```
	cd <path_to_hw1>
	$ pip install -e .
	```
Note: if you run into a `DependencyNotInstalled` error with GLIBC, run `conda install -c conda-forge libstdcxx-ng>=14 libgcc-ng>=14 -y` after creating the conda env and before installing the packages (between step 3 and 4).

This conda environment requires activating it every time you open a new terminal (in order to run code), but the benefit is that the required dependencies for this codebase will not affect existing/other versions of things on your computer. This stand-alone environment will have everything that is necessary.


B. Install on system Python:

	```
		pip install -r requirements.txt 
	   	cd <path_to_hw1> 
	   	pip install -e .
	```

## Common installation errors
Mujoco-py versions above 2 are now deprecated for Windows. We recommend using Linux or running on colab.

### Linux/Ubuntu
If an error appears when installing mujoco-py: `No such file or directory: 'patchelf'`, try installing patchelf first:

```
sudo apt-get install patchelf
```

During mujoco-py install, if `fatal error: GL/osmesa.h: No such file or directory compilation terminated` appears, run:

```
$ sudo apt-get install libosmesa6-dev
```

### Mac
If installing requirements.txt fails to build matplotlib, install it separately without specifying version number:
```
pip uninstall matplotlib
pip install matplotlib
```

### Mac M1
Successfully installing and running Mujoco on Mac M1 may require some additional overhead. Make sure home-brew is [installed correctly](https://blog.smittytone.net/2021/02/07/how-to-migrate-to-native-homebrew-on-an-m1-mac/) and see the installation guides below for help:

https://github.com/openai/mujoco-py/issues/682

https://github.com/openai/mujoco-py/issues/662

Issue installing Cython3: see [this post](https://github.com/openai/mujoco-py/issues/773)