from State import State

class Maze():
    def __init__(self,reward_matrix: list,final_state: list, probabilities: list) -> None:
        self.grid = [[]]
        self.states = []
        self.final_state = final_state
        self.size= (len(reward_matrix[0]),len(reward_matrix))
        self.reward_matrix = reward_matrix
        self.value_matrix = [[]]
        self.probability_matrix = probabilities
        self.action_space = [(1,0),(-1,0),(0,1),(0,-1)]
        #immediately set the initial states
        self.set_initial_states()
        self.set_v_values()
    
    def set_initial_states(self) -> None:
        """sets state objects to Maze class
        Args:
            size (int): shape of reward and state matrix (4,4)
        """
        for i in range(self.size[0]):
            row=[]
            for j in range(self.size[1]):
                "we retrieve reward from reward matrix, on same position as the indexes"
                reward = self.reward_matrix[i][j]
                "we retrieve boolean terminal var, only if position is present in the given positions of the terminal states"
                terminal = (i,j) in self.final_state
                s = State((i,j),reward,terminal)
                row.append(s)
            self.states.append(row)
    
    def set_v_values(self) -> None:
        """initialises all utilities on value 0
        """
        self.value_matrix = [[0 for i in range(self.size[0])] for j in range(self.size[1])]

    def get_states(self) -> list:
        """returns all states of the environment
        Returns:
            list: list of all state objects
        """
        return self.states
    
    def get_values(self) -> None:
        """returns utilities

        Returns:
            _type_: 2D list of all utilities
        """
        return self.value_matrix

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
    
    def get_reward_matrix(self) -> list:
        """Only used to visualise the direct rewards in the pygame window

        Returns:
            list: 2d list of the matrix containing state rewards
        """
        return self.reward_matrix
    
    def get_probability_matrix(self):
        return self.probability_matrix
    
    def update_value_matrix(self, value: float, position: tuple) -> None:
        """update value on current state position after policy has calculated new action

        Args:
            value (float): new calculated utility that needs to be assigend to the right position
            position (tuple): position of new state that represents utility of that new state
        """
        print(f"      update function of maze has started..")
        print(f"      the max utility {value} is updated on position {position}.")
        print(f"      old value matrix: {self.value_matrix}")
        #we need to update de value matrix based on a position (x,y). 
        self.value_matrix[position[0]][position[1]] = value
        print(f"      new value matrix: {self.value_matrix}")
    
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

    
