from FourRooms import FourRooms
import numpy as np

def main():
    scenario = 'simple'
    alpha = 0.5  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 1000

    # Initialize FourRooms object
    fourRoomsObj = FourRooms(scenario)
    #getting the start position of the agent and number of remaining packages
    x, y = fourRoomsObj.getPosition()  # unpack x and y correctly
    
    k = fourRoomsObj.getPackagesRemaining()  # Get remaining packages number
    print("Agent starts at: ({}, {}), with {} packages remaining".format(x, y, k))  # Output the starting position and package count

    # Initialize Q-table: shape will depend on the state representation and number of actions
    Q = np.zeros((11, 11, 2, 4))  # Assuming only k=1 or k=0 and 4 actions

    for episode in range(num_episodes):
        fourRoomsObj.newEpoch()  # Reset environment
        done = False
        while not done:
            x, y = fourRoomsObj.getPosition()
            x = max(0, min(x, 10))
            y = max(0, min(y, 10))

            k = fourRoomsObj.getPackagesRemaining()
            
            state = (x, y, k)
            #print(state)
            # Epsilon-greedy policy
            if np.random.rand() < epsilon:
                action = np.random.choice(4)
            else:
                action = np.argmax(Q[x, y, k])

            # Execute action
            gridType, (new_x, new_y), packagesRemaining, isTerminal = fourRoomsObj.takeAction(action)
            
            new_x = max(0, min(new_x, 10))
            new_y = max(0, min(new_y, 10))

            
            new_state = (new_x, new_y, packagesRemaining)
            reward = -1 if not isTerminal else 100
            
            # Q-value update
            Q[x, y, k, action] += alpha * (reward + gamma * np.max(Q[new_x, new_y, packagesRemaining]) - Q[x, y, k, action])
            
            if isTerminal:
                done = True

        # Optionally decrease epsilon and alpha here over episodes

    # Show final path
    fourRoomsObj.showPath(-1, savefig='final_path_scenario1.png')
    #result = fourRoomsObj.takeAction(action)
    #print(result)
if __name__ == "__main__":
    main()
