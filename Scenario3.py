from FourRooms import FourRooms
import numpy as np

def main():
    scenario = 'rgb'
    alpha = 0.5  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 1  # Exploration rate
    epsilon_min = 0.01  # Minimum exploration rate
    epsilon_decay = 0.995  # Decay multiplier for epsilon
    num_episodes = 500  # Adjust based on convergence needs

    fourRoomsObj = FourRooms(scenario)
    
    Q = np.zeros((11, 11, 4, 4))  # Grid size 11x11, 4 states, 4 actions
    
    state_mapping = {"none": 0, "R": 1, "RG": 2, "RGB": 3}
    package_collection_sequence = ['R', 'G', 'B']

    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()
        done = False
        
        current_package_state = 0 # Starting with 0 packages collected
        while not done:
            x, y = fourRoomsObj.getPosition()  # Get current state
            #state = state_mapping[current_package_state]
            #Clamping to ensure positions are within bounds
            x, y = max(0, min(x, 10)), max(0, min(y, 10))

            # selection of E-greedy action 
            #if np.random.rand() < epsilon:
                #action = np.random.choice(4)
            #else:
                #action = np.argmax(Q[x, y, current_package_state])
                
            action = np.random.choice(4) if np.random.rand() < epsilon else np.argmax(Q[x, y, current_package_state])


            # Execute action
            
            gridType, (new_x, new_y), new_package_collected, isTerminal = fourRoomsObj.takeAction(action)
            #new_state = state_mapping[new_package_state]
            
            new_x, new_y = max(0, min(new_x, 10)), max(0, min(new_y, 10))  # Clamping
            new_package_state = state_mapping.get(new_package_collected, current_package_state)

            
            reward = -1 if not isTerminal else 100
            if new_package_collected and new_package_collected != package_collection_sequence[current_package_state]:
                reward = -100  #  penalty
                isTerminal = True
            #else:
                #reward = -1 if not isTerminal else 100
                
            print(f"Episode {episode}, Position ({x}, {y}), State {current_package_state}, Action {action}, New State {new_package_state}, Reward {reward}")

            # Update Q-value
            Q[x, y, current_package_state, action] += alpha * (reward + gamma * np.max(Q[new_x, new_y, new_package_state]) - Q[x, y, current_package_state, action])

            # Update current package state
            current_package_state = new_package_state

            if isTerminal:
                done = True
                
        epsilon = max(epsilon_min, epsilon * epsilon_decay)
            #print(f"Episode {episode}: Epsilon {epsilon}")

    fourRoomsObj.showPath(-1, savefig='final_path_scenario3.png')

if __name__ == "__main__":
    main()
