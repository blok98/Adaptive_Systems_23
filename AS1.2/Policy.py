from State import State

import random

class Policy():
    def __init__(self) -> None:
        self.size=(0,0)
    
    def set_size(self,size: tuple):
        #set size to prevent going out of bounds
        self.size=size

    def select_action(self,action_space: list):
        #for now chose a random action
        #but make sure to not go out of bounds
        #its defined as (n_column,n_row). so (2,3) is 3 right and 4 under
        action = random.choice(action_space)
        return action

    
