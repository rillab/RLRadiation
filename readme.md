# Reinforcement learning for Radiation Source Detection
By Zheng Liu and Shiva Abbaszadeh (NPRE, UIUC)

Current radiation source survey strategies either require human efforts or are not efficient and flexible enough to adjust survey paths based on recent measurements. Reinforcement learning, which studies the problem of how agents ought to take the optimized action so that a goal can be achieved efficiently, provides an alternative data-driven solution to conduct radiation detection tasks with no human intervention. This code provides the simulated radiation environment and a double Q-learning algorithm for automated source searching. This work was initially described in this [paper](https://www.mdpi.com/1424-8220/19/4/960).

### Citing this work
If you find this work useful in your research, please consider citing:

	@article{liu2019double,
	  title={Double Q-Learning for Radiation Source Detection},
	  author={Liu, Zheng and Abbaszadeh, Shiva},
	  journal={Sensors},
	  volume={19},
	  pages={960},
	  year={2019}
	}

### Files/folders:
    
   * `./experiments/`: folder to store training process.   
   * `simulation.py`: Python module to build a simulation environment for the radiation source detection problem. Reinforcement learning algorithm is trained in this simulation environment.
   * `reinforcement_learning.py`: Python module to train the reinforcement learning algorithm. This algorithm is implemented with  Tensorflow. 
   * `results_visualization.ipynb`: The ipython notebook to illustrate the training result of the reinforcement learning algorithm.

Dependences: Python3, tensorflow, numpy, matplotlib

### How to use:
1. Model training
- Make sure the root directory has `simulation.py`, `reinforcement_learning.py`, and ` ./experiments/`.
- To train the model, run: `$ python3 reinforcement_learning.py`
- The trained model will be automatically saved in `./experiments/`
2. Results visualization:
        Open the `results_visualization.ipynb`, and run corresponding cells.

