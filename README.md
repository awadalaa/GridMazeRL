# GridWorld Q-Learning Demo Webapp

This project is a simple web application that demonstrates the basics of reinforcement learning using Q-learning in a gridworld environment. An agent learns to navigate a 5×5 grid from a starting point at (0, 0) to a goal at (4, 4) by updating its Q-values based on the rewards it receives for each move.

## Features

- **Simple Environment:** A 5×5 grid with a clear start (top-left) and goal (bottom-right).
- **Reinforcement Learning:** Implementation of a basic Q-learning algorithm with an ε-greedy exploration strategy.
- **Interactive Web Interface:** A minimal web page to visualize the grid, the agent's position, and control the learning process with step and reset buttons.
- **Educational Purpose:** Ideal for beginners who want to understand the core concepts of reinforcement learning and web development integration.

## Getting Started

### Prerequisites

- Python 3.6 or higher
- Flask

Install Flask using pip:

```bash
pip install flask
```

### Running the Application
**Run the Flask Server:**

In the project directory, run:

```bash
python -m app
```

**Open in Browser:**
Open your web browser and go to http://127.0.0.1:5000/ to view and interact with the webapp.

Project Structure
`app.py`: The Flask server that handles the backend logic. It defines the GridWorld environment, the Q-learning agent, and the API endpoints (/reset and /step).

`templates/index.html`: The front-end HTML file that renders the grid, displays the agent's current position, and provides controls (Step and Reset) for interacting with the simulation.

### How It Works
**Environment Setup:**

* The grid is defined as a 5×5 matrix.
* The starting position is (0, 0) and the goal is (4, 4).
* The GridWorld class manages the environment, resetting the state and processing moves.

**Q-Learning Agent:**
The agent uses Q-learning to decide which action to take (up, down, left, or right).
The Q-values are stored in a table (a Python dictionary) and updated after every move using the learning rate, discount factor, and rewards.
Web Interface:

* The grid is rendered as an HTML table where the agent is denoted by "A" and the goal by "G".
* The Step button sends a request to the /step endpoint, allowing the agent to take a move and learn from it.
* The Reset button sends a request to the /reset endpoint, restarting the environment.

### Learning Outcomes
Gain hands-on experience with reinforcement learning concepts such as states, actions, rewards, and the exploration/exploitation trade-off.
Understand how to integrate a machine learning algorithm with a web application.
Learn basic web development using Flask, HTML, CSS, and JavaScript.

### Future Enhancements
* Add obstacles or more complex terrain to the grid.
* Visualize the Q-table or track performance metrics in real time.
* Allow dynamic adjustment of learning parameters (e.g., learning rate, discount factor, ε).
* Extend the environment to support more complex scenarios.

### License
This project is open-source and available under the MIT License.

### Acknowledgments
This demo was created as an educational example for those starting with reinforcement learning and web development. Contributions and suggestions for improvements are welcome!
