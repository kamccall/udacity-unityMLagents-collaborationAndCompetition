# udacity-unityMLagents-collaborationAndCompetition
udacity collaboration and competition programming assignment learning to control the rackets of players in a tennis game in order to volley a ball back and forth over a net 

# Project Details 
In this project, we trained a DRL agent with the Unity MLAgents environment to teach tennis players how to control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. The goal of each agent is to keep the ball in play.  The environment is considered �solved� when the maximum score is above 0.5 between the two players for the trailing 100 consecutive episodes.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. The last three observations are stacked, so the observation space vector actually consists of 24 floating point numbers. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.  The task is episodic, with a maximum number of timesteps of 1000. 

# Getting Started
The project is comprised of three files:
1. **tennis.ipynb**: imports packages, connects to Unity environment, and then runs the learning function that will interact with that environment, leveraging various neural networks defined in model.py and the Agent class defined in ddpg_agent.py
2. **model.py**: contains the pytorch neural networks (two actor networks, and two critic networks), that actively learn and store the learnings from the DDPG training process  
3. **ddpg_agent.py**: contains the code to initialize the (four) networks, a replay buffer class, an OUNoise class, and wrappers for selecting actions, submitting them to the environment (through 'step') and then learn from experience stored in the replay buffer

Logically, there are three steps required in order to execute the project: Download the files referred to above, install the necessary Unity MLAgents environment, and execute the code in the environment of your choice, such as a Jupyter Notebook environment (which is recommended for simplicity).  Here are the more detailed instructions for doing so:
1. Download the three files above (from the public github repo)   
Execute this command (at a shell prompt, in the appropriate source repo location where you wish to copy the files): \
$ *git clone https://github.com/kamccall/udacity-unityMLagents-collaborationAndCompetition* \
This will clone the entire repository - including the three source files as well as associated README and REPORT files - into that source directory. 
1. Install the Unity MLAgents environment (within which the agent will train and subsequently execute)   
Follow the directions below in order to install the needed 'Tennis' environment for Unity MLAgents: \
https://github.com/udacity/deep-reinforcement-learning/tree/master/p3_collab-compet#getting-started
1. Execute the code (either within a Jupyter Notebook environment, or within an alternative IDE) \
There are dependencies upon various Python packages in order to successfully execute this project, such as numpy, pyplot, torch, and others. Depending upon your Python environment, it is likely that those packages are already installed, in which case the 'import' commands in the code will successfully execute, thereby allowing the use of the code in those packages.\
If your environment does not already have all of these packages installed, it is possible that you will need to install one or more packages before you can successfully execute the 'import' command to utilize them.  If that is the case:
    1. ADD a new cell at the top of your Jupyter notebook
    1. Follow the appropriate directions in this link to install the required packages (which will be different depending upon whether you are using 'pip' or 'conda' as your package manager): https://jakevdp.github.io/blog/2017/12/05/installing-python-packages-from-jupyter/

# Build and Test Instructions
There are three ways to run the code:
1. Install all three files in the same directory, and execute the navigation .ipynb notebook file in a Jupyter notebook server , or
2. Insert the code from 'model.py' and 'ddpg_agent.py' into new (inserted) cells in the .ipynb notebook file, and execute everything from the single notebook file within a Jupyter notebook environment, or
3. (More difficult): Copy and paste the Python code from within the .ipynb notebook file into a Python source code file, and then execute the program from the command line (with all appropriate dependencies based on your IDE and OS)

# Attribution
The general approach used in creation of this project, just like the last project (which mostly influenced the ddpg_agent.py and .ipynb files) was from the DDPG implementation of BipedalWalker-v2 here: \
https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-bipedal \
In addition, after I had working code with the last project (continousControl) - but an agent that wasn't learning - I knew something was wrong in the 'learn' method of the agent. I reviewed DDPG implementations of both Gym and Unity environments, and stumbled on an important line of code (that performs gradient clipping when training the critic) from this repo: \
https://github.com/AkshayS21/Reacher-Environment-Continuous-Control-with-DDPG-algorithm
