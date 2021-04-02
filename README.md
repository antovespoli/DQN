# DQN
Deep Q-Network implementation with replay buffer 

For this coursework, I implemented a DQN for solving a maze where an agent has to reach a goal. From a high-level perspective, the agent learns from its experience. After every step, a positive or negative reward is collected depending on the current state and the new state (lines 254-263). In particular, the agent is positively rewarded for getting very close to the goal (line 254), punished for hitting a wall (line 257), and positively rewarded for effectively moving to the right (line 260). Moving vertically is neither punished nor rewarded (line 263). Also, hitting a wall is punished proportionally to the distance from the goal.
At every step, the agent can choose between three possible actions: up, down, right (240-248). During training, deciding what action will be taken is controlled by an ε greedy policy (lines 196-211), where the best action is selected with probability (1 − ε) + ε/3 and all the other actions each with probability ε/3. This allows the agent to both explore the environment and exploit what it has learnt so far. In order to optimize this trade-off between exploration and exploitation, I have implemented an ε decay (lines 230-237) that depends on the number of steps taken so far and is clipped at 0.005. Specifically, ε is decayed slowly for the first 30,000 steps and then faster (line 233). This is to allow better exploration at the beginning and to maximize exploitation later on in the learning process.
As the agent experiences the environment, it collects observations about the tran- sitions it makes (line 265). Those transitions are stored by a prioritised experience replay buffer (lines 78-130). The buffer has a capacity of 100,000 transitions (line 162). Once full, the oldest observations get overwritten (line 116). After every step, a batch of observations is randomly sampled from the buffer (lines 28-50). The probability of each observation to be sampled depends on its weight and is guaranteed to be non zero (min prob = 10−6) (lines 92-94). The episodes are initially set to be 250 steps long (line 135), however, after 10,000 steps, the length of the episodes starts to be linearly decreased by 25 steps after every 10 episodes (line 177). Episodes are bounded to have at least 100 steps (line 177). The choice of progressively reducing the episode length is to speed up the process later on in the learning when the agent starts to exploit more the gained knowledge and to avoid wasting computational time. Indeed, since there is no left action, once the rightmost wall has been reached, the agent can’t perform any useful exploration anymore. Every 21 episodes, a long episode of 500 steps is performed to allow for a deeper exploration of the maze (lines 184-186).
In conclusion, the overall training process is the following: the agent moves around according to an ε greedy policy (lines 196-211) collecting observations that get stored in a prioritised replay buffer (lines 78-130). After every step, 128 transitions in the buffer are sampled (line 222) and used to train the Q-network (line 223). Every 50 steps, the target network gets updated by copying the weights of the Q-network (lines 74-76, 277). After performing 10,000 steps, the greedy policy gets evaluated every 10 episodes (lines 179-182, 213-217). If the greedy policy is found to reach the area nearby the goal, the flag done is set to True and no further training is performed (line 284).
