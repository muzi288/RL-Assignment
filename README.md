# Reinforcement Learning in Four Rooms Environment

This repository contains implementation of various scenarios using a Reinforcement Learning (RL) model in a simulated environment called "Four Rooms". The environment is designed to test different RL strategies and scenarios.

## Project Structure

The project includes multiple scenarios, each represented by a separate Python script. For each scenario, there is also a corresponding script that includes stochastic actions to introduce unpredictability in the agent's movement.

## NOTE FOR SCENARIO 4!
scenario 4 was implement by creating separate python scripts for each scenario. this was done to avoid messing up the original code for each scenario should bugs occur while implementing stochastic actions.
THESE FILES ARE CALLED Scenario1Stochatics.py and Scenario2Stochastic.py

### Files and Descriptions

- `FourRooms.py`: The core module that defines the Four Rooms environment.
- `Scenario1.py`: Implements the basic RL scenario.
- `Scenario1Stochatics.py`: Implements the basic RL scenario with stochastic actions.
- `Scenario2.py`: Implements a multi-goal RL scenario.
- `Scenario2Stochastic.py`: Implements the multi-goal RL scenario with stochastic actions.
- `Scenario3.py`:Agent collects 3 packages located somewhere in the environment. These packages are marked as red (R), green (G) and blue (B) and must be collected in that order.
- `requirements.txt`: Lists all Python dependencies for the project.

## Setup and Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/four-rooms-rl.git
   cd four-rooms-rl
