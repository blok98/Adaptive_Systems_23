from State import State

class Maze():
    def __init__(self,reward_matrix: list,final_state: list) -> None:
        self.grid = [[]]
        self.states = []
        self.final_state = final_state
        self.reward_matrix = reward_matrix
        self.size= (len(reward_matrix[0]),len(reward_matrix))
        self.action_space = [(1,0),(-1,0),(0,1),(0,-1)]
        #immediately set the initial states
        self.set_initial_states(size=self.size)
    
    def set_initial_states(self,size: tuple) -> None:
        """sets state objects to Maze class

        Args:
            size (int): shape of reward and state matrix (4,4)
        """
        for i in range(size[0]):
            row=[]
            for j in range(size[1]):
                "we retrieve reward from reward matrix, on same position as the indexes"
                reward = self.reward_matrix[j][i]
                "we retrieve boolean terminal var, only if position is present in the given positions of the terminal states"
                terminal = (i,j) in self.final_state
                s = State((i,j),reward,terminal)
                row.append(s)
            self.states.append(row)
    
    def get_states(self) -> list:
        """returns all states of the environment

        Returns:
            list: list of all state objects
        """
        return self.states
    
    def get_size(self) -> tuple:
        """Returns the shape of the environment

        Returns:
            tuple: shape of amount of rows and amount of columns
        """
        return self.size
    
    def get_actionspace(self) -> list:
        """Returns list of all possible actions an agent can make, without taking in account of edges

        Returns:
            list: list of tuples with delta coordinates
        """
        return self.action_space

    def get_terminal_states(self) -> list:
        """Returns terminal state positions, required for Agent to know when to stop

        Returns:
            list: the coordinates of the final states as a tuple
        """
        return self.final_state
    
    def step(self,current_position: tuple, action: tuple) -> State:
        """makes the agent take an action - moving to another cell

        Args:
            current_position (tuple): current position of the current state the agent is in
            action (tuple): the action that is decided to take by the agent

        Returns:
            State: the new state after taking the given action
        """
        new_position = (current_position[0]+action[0],current_position[1]+action[1])
        print(f"old position: {current_position}, action: {action}, new position: {new_position}, ",end="")
        new_state = self.states[new_position[0]][new_position[1]]
        print(f"new state: {new_state}")
        #we only return the new state, all values (pos,reward) are just attribute to that state
        return new_state

    
