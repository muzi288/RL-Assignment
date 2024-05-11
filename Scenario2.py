from FourRooms import FourRooms
import numpy as np

def main():
    scenario = 'multi'
    alpha = 0.5  # Learning rate
    gamma = 0.9  # Discount factor
    epsilon = 0.1  # Exploration rate
    num_episodes = 500  # Modify as necessary

    fourRoomsObj = FourRooms(scenario)
