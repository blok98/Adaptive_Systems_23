class State():
    def __init__(self, position: tuple, reward: int, terminal: bool) -> None:
        self.position = position
        self.reward = reward
        self.terminal = terminal
    
    def get_position(self) -> tuple:
        """returns position of the state, needed for agent to get information and to calculate new position

        Returns:
            tuple: tuple of the coordinates of the state
        """
        return self.position
    
    def __str__(self):
        return f"State( {self.position}, {self.reward}, {self.terminal} )"