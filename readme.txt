Project: Reinforcement learning for Radiation Source Detection
Author: Zheng Liu
Email: zliu86@illinois.edu
Date: May 6, 2019

Files/folders:
    
    ./experiments/: folder to store training process.
    
    simulation.py: Python module to build a simulation environment for the radiation source detection problem. Reinforcement learning algorithm is trained in this simulation environment.
    
    reinforcement_learning.py: Python module to train the reinforcement learning algorithm. This algorithm is implemented with  Tensorflow. 
        
    results_visualization.ipynb: The ipython notebook to illustrate the training result of the reinforcement learning algorithm.

Dependences: Python3, tensorflow, numpy, matplotlib

How to use:
    1. Model training
        Make sure the root directory has:
            simulation.py
            reinforcement_learning.py
            ./experiments/
        To train the model, just run: 
            $ python3 reinforcement_learning.py
            The trained model will be automatically saved in ./experiments/
    2. Results visualization:
        Open the results_visualization.ipython, and run corresponding cells.

