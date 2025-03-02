from flask import Flask, render_template, jsonify, request
import random

app = Flask(__name__)

GRID_SIZE = 5
ACTIONS = ['up', 'down', 'left', 'right']

# Extended GridWorld environment with mode support
class GridWorld:
    def __init__(self, grid_size=GRID_SIZE, mode="simple"):
        self.grid_size = grid_size
        self.start = (0, 0)
        self.goal = (grid_size - 1, grid_size - 1)
        self.mode = mode
        self.state = self.start

        if mode == "simple":
            self.obstacles = [(1, 1), (1, 2), (2, 3)]
            self.reward_zones = []
            self.penalty_zones = []
        elif mode == "complex":
            self.obstacles = [(1, 1), (1, 2), (2, 3), (3, 0), (3, 1)]
            self.reward_zones = [(0, 4)]   # stepping here gives a moderate reward
            self.penalty_zones = [(2, 2)]    # stepping here gives a penalty
        else:
            self.obstacles = []
            self.reward_zones = []
            self.penalty_zones = []

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

        # Check for obstacles: if encountered, remain in place and penalize heavily
        if new_state in self.obstacles:
            reward = -5
            new_state = self.state
            done = False
        else:
            reward = -1  # default move penalty
            done = False
            # In complex mode, check additional zones
            if self.mode == "complex":
                if new_state in self.reward_zones:
                    reward = 5
                elif new_state in self.penalty_zones:
                    reward = -3

            if new_state == self.goal:
                reward = 10
                done = True

        self.state = new_state
        return new_state, reward, done

# Q-Learning agent
class QLearningAgent:
    def __init__(self, actions=ACTIONS, learning_rate=0.1, discount_factor=0.95, epsilon=0.2):
        self.q_table = {}  # dictionary: (state, action) -> value
        self.actions = actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon

    def get_q(self, state, action):
        return self.q_table.get((state, action), 0.0)

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.choice(self.actions)
        else:
            q_values = [self.get_q(state, a) for a in self.actions]
            max_q = max(q_values)
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

# Instantiate environment and agent with default parameters and mode
env = GridWorld(mode="simple")
agent = QLearningAgent()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/reset', methods=['POST'])
def reset():
    data = request.get_json() or {}
    mode = data.get("mode", "simple")
    # Reinitialize the environment with the given mode and clear the Q-table.
    global env, agent
    env = GridWorld(mode=mode)
    agent.q_table = {}
    state = env.reset()
    return jsonify({'state': state, 'obstacles': env.obstacles, 'mode': env.mode})

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
        'q_table': {str(k): v for k, v in agent.q_table.items()}
    })

@app.route('/update_params', methods=['POST'])
def update_params():
    data = request.get_json()
    try:
        agent.learning_rate = float(data.get("learning_rate", agent.learning_rate))
        agent.discount_factor = float(data.get("discount_factor", agent.discount_factor))
        agent.epsilon = float(data.get("epsilon", agent.epsilon))
        return jsonify({"status": "success", "learning_rate": agent.learning_rate,
                        "discount_factor": agent.discount_factor, "epsilon": agent.epsilon})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
