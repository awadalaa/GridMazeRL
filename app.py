from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

# Define grid size and possible actions
GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right']

# GridWorld environment
class GridWorld:
    def __init__(self, grid_size=GRID_SIZE):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.state = self.start

    def reset(self):
        self.state = self.start
        return self.state

    def step(self, action):
        x, y = self.state
        if action == 'up':
            new_state = (max(x - 1, 0), y)
        elif action == 'down':
            new_state = (min(x + 1, self.grid_size - 1), y)
        elif action == 'left':
            new_state = (x, max(y - 1, 0))
        elif action == 'right':
            new_state = (x, min(y + 1, self.grid_size - 1))
        else:
            new_state = self.state

        # Set a default small penalty for each move
        reward = -1
        done = False

        # Check if goal is reached
        if new_state == self.goal:
            reward = 10  # Reward for reaching the goal
            done = True

        self.state = new_state
        return new_state, reward, done

# Q-Learning agent
class QLearningAgent:
    def __init__(self, actions=ACTIONS, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        self.q_table = {}  # Dictionary to store Q-values: key=(state, action)
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        # ε-greedy strategy: explore with probability ε, else choose best known action.
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.get_q(state, a) for a in self.actions]
            max_q = max(q_values)
            # In case of ties, randomly choose among best actions.
            best_actions = [a for a, q in zip(self.actions, q_values) if q == max_q]
            return random.choice(best_actions)

    def learn(self, state, action, reward, next_state, done):
        current_q = self.get_q(state, action)
        if done:
            target = reward
        else:
            next_q_values = [self.get_q(next_state, a) for a in self.actions]
            target = reward + self.discount_factor * max(next_q_values)
        self.q_table[(state, action)] = current_q + self.learning_rate * (target - current_q)

# Instantiate environment and agent
env = GridWorld()
agent = QLearningAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    state = env.reset()
    return jsonify({'state': state})

@app.route('/step', methods=['POST'])
def step():
    state = env.state
    action = agent.choose_action(state)
    next_state, reward, done = env.step(action)
    agent.learn(state, action, reward, next_state, done)
    return jsonify({
        'state': next_state,
        'action': action,
        'reward': reward,
        'done': done,
        # For debugging: you can view the Q-table updates
        'q_table': {str(k): v for k, v in agent.q_table.items()}
    })

if __name__ == '__main__':
    app.run(debug=True)

