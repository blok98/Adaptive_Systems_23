from State import State

import random

class Policy():
    def __init__(self) -> None:
        self.size=(0,0)
    
    def set_size(self,size: tuple) -> None:
        """set size to prevent going out of bounds

        Args:
            size (tuple): the shape of the grid, used for taking action
        """
        self.size=size

    def select_action(self,action_space: list) -> tuple:
        """ For now chose a random action
        but make sure to not go out of bounds
        its defined as (n_column,n_row). so (2,3) is 3 right and 4 under

        Args:
            action_space (list): list of all possible actions, limited by position

        Returns:
            tuple: chosen random action
        """
        action = random.choice(action_space)
        return action

    
