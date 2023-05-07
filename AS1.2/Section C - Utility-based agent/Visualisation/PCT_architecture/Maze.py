from State import State

class Maze():
    def __init__(self,reward_matrix: list,final_state: list) -> None:
        self.grid = [[]]
        self.states = []
        self.final_state = final_state
        self.size= (len(reward_matrix[0]),len(reward_matrix))
        self.reward_matrix = reward_matrix
        self.value_matrix = [[]]
        self.action_space = [(1,0),(-1,0),(0,1),(0,-1)]
        #immediately set the initial states
        self.set_initial_states()
        self.set_v_values()
    
    def set_initial_states(self) -> None:
        #first we safe all states on position(j,i) in self.states
        #we give state position (j,i) so all rows are initialised first and saved in states matrix
        for i in range(self.size[0]):
            row=[]
            for j in range(self.size[1]):
                "we retrieve reward from reward matrix, on same position as the indexes"
                reward = self.reward_matrix[j][i]
                "we retrieve boolean terminal var, only if position is present in the given positions of the terminal states"
                terminal = (i,j) in self.final_state
                s = State((i,j),reward,terminal)
                row.append(s)
            self.states.append(row)
    
    def set_v_values(self) -> None:
        '''
        sets 
        '''
        self.value_matrix = [[0 for i in range(self.size[0])] for j in range(self.size[1])]
    
    def set_env(self,size: int) -> None:
        #set up a grid with size = x*x, given parameter size=(x,x)
        self.grid=[[-1]*size[0]]*size[1]

    def get_states(self) -> None:
        return self.states
    
    def get_values(self) -> None:
        '''
         Agent has no acces to this, because the agent only looks one step forward.
         '''
        return self.value_matrix

    def get_size(self) -> tuple:
        return self.size
    
    def get_actionspace(self) -> list:
        return self.action_space

    def get_terminal_states(self) -> list:
        '''
        returns terminal state positions, required for Agent to know when to stop
        '''
        return self.final_state
    
    def get_reward_matrix(self):
        '''
        Only used to visualise the direct rewards in the pygame window. 
        Agent has no acces to this, because the agent only looks one step forward.
        '''
        return self.reward_matrix
    
    def get_state_info(self,position: tuple) -> tuple:
        '''
        returns reward and utility of state (a,b)
        '''
        return self.reward_matrix[position[0]][position[1]], self.value_matrix[position[0]][position[1]]
    
    def update_value_matrix(self, value: float, position: tuple) -> None:
        "update value on current state position after policy has calculated new action"
        self.value_matrix[position[0]][position[1]] = value
    
    def step(self,current_position: tuple, action: tuple) -> tuple:
        #makes the agent take an action - moving to another cell
        new_position = (current_position[0]+action[0],current_position[1]+action[1])
        print(f"old position: {current_position}, action: {action}, new position: {new_position}, ",end="")
        new_state = self.states[new_position[0]][new_position[1]]
        print(f"new state: {new_state}")
        #we only return the new state, all values (pos,reward) are just attribute to that state
        return new_state

    
