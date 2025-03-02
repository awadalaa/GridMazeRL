# Understanding Q-Learning

Q-Learning is a model-free reinforcement learning algorithm that allows an agent to learn an optimal policy for decision making by interacting with an environment. It learns the value (or quality) of each action in a given state, guiding the agent to select actions that maximize cumulative rewards over time.

## How Q-Learning Works

### 1. Initialization
- **Q-Table:**  
  The agent starts with a Q-table, which stores a Q-value for every state-action pair. Initially, these values can be set to zero.
- **Key Parameters:**
  - **Learning Rate (α):**  
    Determines how much new information overrides old information.
  - **Discount Factor (γ):**  
    Balances the importance of future rewards against immediate rewards.
  - **Exploration Rate (ε):**  
    Controls the trade-off between exploring new actions and exploiting the best known actions.

### 2. Interaction with the Environment
- **State Observation:**  
  The agent observes its current state.
- **Action Selection (ε-greedy):**  
  - With probability ε, the agent selects a random action (exploration).  
  - With probability 1-ε, it selects the action with the highest Q-value for the current state (exploitation).
- **Taking the Action:**  
  The chosen action is executed, and the agent transitions to a new state.
- **Receiving a Reward:**  
  The environment provides a reward based on the action taken.

### 3. Q-Value Update
After taking an action, the Q-value for the state-action pair is updated using the following formula:

