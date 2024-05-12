from FourRooms import FourRooms
import numpy as np
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the Four Rooms environment with optional stochastic actions for Scenario 1.")
    parser.add_argument("-stochastic", action="store_true", help="Enable stochastic action space.")
    args = parser.parse_args()
    return args





def main():
    
    args = parse_arguments()
    scenario = 'multi'
    alpha = 0.5  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 500  # Modify as necessary
    stochastic = args.stochastic

    #fourRoomsObj = FourRooms(scenario)
    fourRoomsObj = FourRooms(scenario, stochastic=stochastic)
    
    Q = np.zeros((11, 11, 8, 4))  # Adjusted size as here are 2^3=8 possible states for the packages
    
    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        done = False
        while not done:
            x, y = fourRoomsObj.getPosition()  
            package_state = fourRoomsObj.getPackagesRemaining()
            #state_index = sum([2**i for i, has_package in enumerate(package_state) if has_package])
            
            x = max(0, min(x, 10))
            y = max(0, min(y, 10))

            # E-greedy policy to choose action
            if np.random.rand() < epsilon:
                
                action = np.random.choice(4)
            else:
                #action = np.argmax(Q[x, y, state_index])
                action = np.argmax(Q[x, y, package_state])

            # Take action and observe results
            gridType, (new_x, new_y), new_package_state, isTerminal = fourRoomsObj.takeAction(action)
            #new_state_index = sum([2**i for i, has_package in enumerate(new_package_state) if has_package])
            new_x = max(0, min(new_x, 10))
            new_y = max(0, min(new_y, 10))

            
            reward = -1 if not isTerminal else 100

            # Update Q-table
            #Q[x, y, state_index, action] += alpha * (reward + gamma * np.max(Q[new_x, new_y, new_state_index]) - Q[x, y, state_index, action])
            Q[x, y, package_state, action] += alpha * (reward + gamma * np.max(Q[new_x, new_y, new_package_state]) - Q[x, y, package_state, action])
            
            if isTerminal:
                done = True

    fourRoomsObj.showPath(-1, savefig='final_path_scenario2.png')

if __name__ == "__main__":
    main()


