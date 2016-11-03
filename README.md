Implement tic-tac-toe using deep q network. The agent learns both the rules and the strategy of the game from experience. To force the agent to learn the rules, I apply a heavy penalty for cheating ( placing a move on an illegal spot )

- game.py: implement a simple tic tac toe game environment
- train.py: driver for gathering experiences and training
- rl/deep_q_network.py: implementation for deep q network

# Network Architecture:

A simple feedforward two hidden layer network

# Reward Structure:

- Won: 100
- Draw: 10
- Lost: -1
- Cheating (placing a move on a taken spot): -10

Future rewards are discounted

# Training Parameters
- Initial exploration epsilon: 0.6
- Final exploration epsilon: 0.1
- Discount factor: 0.8
- Regularization strength: 0.01
- Target network update rate: 0.01

# Experiments

The following shows the average reward from last 100 games that have been played over a training period of about 180k games. Orange is when the agent plays against a random player. Yellow is when the agent plays against a near-optimal strategy player. In both cases, the agent always makes the first move

![Alt text](/screenshots/game_reward.png?raw=true&style=centerme "Experiments")
