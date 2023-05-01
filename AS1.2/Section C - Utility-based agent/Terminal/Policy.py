from State import State
from Agent import Agent

import random

class Policy():
    def __init__(self) -> None:
        self.size=(0,0)
    
    def set_size(self,size: tuple):
        #set size to prevent going out of bounds
        self.size=size

    def get_neighbour_values(self,action_space: list, current_state: State, state_matrix: list, value_matrix: list):
        current_position = current_state.get_position()
        nextstates_rewards = []
        for action in action_space:
            new_position = (current_position[0]+action[0],current_position[1]+action[1])
            nextstates_reward = state_matrix[new_position[0]][new_position[1]].get_reward()
            nextstates_rewards.append(nextstates_reward)
        return nextstates_rewards

    def select_action(self,action_space: list, current_state: State, state_env: list, value_function):
        #for now chose a random action
        #but make sure to not go out of bounds
        #its defined as (n_column,n_row). so (2,3) is 3 right and 4 under

        #get neighbourvalues based on actionspace, ordered by actions in actionspace
        nextstates_rewards = self.get_neighbour_values()
         
        action = random.choice(action_space)
        return action

    
