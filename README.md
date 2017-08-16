## PongReinforcementLearning
Deep DQN Based Reinforcement Leaning for simple Pong PyGame 

This DQN experiment plays a Py Pong Game (DQN control of Left Hand Yellow Paddle)

![alt text](https://github.com/JulesVerny/PongReinforcementLearning/blob/master/PongPicture2.PNG "Game Play")
The Objective is simply measured as succesfully returning of the Ball 
The programmed oponent player is a pretty hot player. Imagine success as being able to return ball served from Serena Williams.
The Moving Average Score is calculated in range from [-10, +10] from Complete failure to return the balls, to full success in returning the Ball. This experiment demonstrates DQN based Reinforcement Learning Agent, which improves from poor performace ~ -8.0 towards reasonably good ~ +9.0 (Fluctuating) return rate in around 10,000 game update cycles [Not returns] The Agent employs Direct Features [ Paddle Y, Ball X, Y and Ball X,Y Directions feeding into DQN Nueral net Estimator of Q[S,A] function. 

This is NOT a Convolutional Network based RL (based Game Video Frame states, because in my experience CN's takes much too Long to Learn on standard PCs: GPUs] and so unfortunately this is Game Specific DQN Reinforecment Learning, and cannot be generalised to other games like Deep minds ATARI game approach 

![alt text](https://github.com/JulesVerny/PongReinforcementLearning/blob/master/ScoreGrowth.png "Score growth")      

### Acknowledgments:

The  Pong Game Code is based upon Siraj Raval's inspiring vidoes on Machine learning and Reinforcement Learning [ Which does employ full convolutional DQN example]  https://github.com/llSourcell/pong_neural_network_live

The DQN Agent Software is Based upon Jaromir Janisch source code: 
https://jaromiru.com/2016/10/03/lets-make-a-dqn-implementation/
