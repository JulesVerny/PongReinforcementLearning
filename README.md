PongReinforcementLearning
Deep DQN Based Reinforcement Leaning for simple Pong PyGame 

MyPong DQN Reinforcement Learning Experiment

Plays Pong Game (DQN control of Left Hand Yellow Paddle)
The Objective is simply measured as succesfully returning of the Ball 
The programed oponent player is a pretty hot player. Imagine success as being able to return ball served from Serena Williams.
The Moving Average Score is calculated in range from [-10, +10] from Complete failure to return the balls, to full success in returning the Ball. This experiment demonstrates DQN based Reinforcement Laerning Agent, improves from poor performace ~ -5.0 towards reasonably good +8.0 (Fluctuating) return rate in around 15,000 game cycles [Not returns] The Agent employs Direct Features [ Paddle Y, Ball X, Y and Ball X,Y Directions feeding into DQN Nueral net Estimator of Q[S,A] function. 

This is NOT a Convolutional Network based RL, based Game Video Frame states [Which in my experience takes much too Long to Learn on standard PCs] and so unfortunaly this is Game Specific DQN Reinforecment Learning, and cannot be generalised to other games. Requires specific Features to be identified. 
      
The  Pong Game Code is based upon Siraj Raval's inspiring vidoes on Machine learning and Reinforcement Learning [ Which is full convolutional DQN example]  https://github.com/llSourcell/pong_neural_network_live

The DQN Agent Software is Based upon Jaromir Janisch  source code: 
# https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
