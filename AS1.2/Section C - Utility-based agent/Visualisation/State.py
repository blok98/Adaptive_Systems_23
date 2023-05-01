class State():
    def __init__(self, position: tuple, reward: int, terminal: bool) -> None:
        self.position = position
        self.reward = reward
        self.terminal = terminal
    
    def get_position(self):
        return self.position
    
    def get_reward(self):
        return self.reward
    
    def __str__(self):
        return f"State( {self.position}, {self.reward}, {self.terminal} )"