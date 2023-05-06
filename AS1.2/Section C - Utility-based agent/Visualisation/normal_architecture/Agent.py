from Maze import Maze
from State import State
import random

class Policy():
    def __init__(self) -> None:
        self.size=(0,0)
    
    def set_size(self,size: tuple):
        #set size to prevent going out of bounds
        self.size=size

    def select_action(self,action_space: list, current_state: State, state_env: list, value_matrix: list, value_function):
        #for now chose a random action
        #but make sure to not go out of bounds
        #its defined as (n_column,n_row). so (2,3) is 3 right and 4 under
        print(f"   policy function select_action started..")
        current_position = current_state.get_position()
        #we define the initial max_value as low as possible 
        max_value = -9999
        best_action = (0,0)
        for action in action_space:
            #calculate position of possible next state by adding the action (0,1) to the current state (2,3)
            nextstate_position = (current_position[0]+action[0],current_position[1]+action[1])
            #define indexes as i,j for shorter code
            i,j = nextstate_position
            # first we retrieve the reward of the next state ( r(s') )
            nextstate_reward = state_env[i][j].get_reward()
            # then we retrieve the value of the next state ( v(s') )
            nextstate_value = value_matrix[i][j]
            print(f"   check action {action}.. nexstatereward = {nextstate_reward}, nextstatevalue = {nextstate_value}, both on position {i,j}.")
            # at last we update current state value with the bellman expectation equation
            currentstate_value = value_function(nextstate_reward, nextstate_value, 1)
            if currentstate_value>=max_value: 
                max_value=currentstate_value
                best_action=action
        print(f"   best action of policy is {best_action} with value {max_value}")

        return best_action, max_value

class Agent():
    def __init__(self, grid: Maze, policy: Policy) -> None:
        self.env=grid
        self.policy = policy
        self.current_state = None
        self.value_func = self.bellman
        self.maze = grid
        #we define start_position in order to reset the agent in the environment when terminal state is reached
        self.start_position = (0,0)
        #define empty sample, which contains all states visited by agent
        self.sample = []
        #we immediately give the policy the maze size to prevent going out of bounds
        policy.set_size(grid.get_size())
    
    def set_current_state(self, position: tuple):
        #set current state based on given position of initial state (maze.states are indexed based on position, so we can do an easy lookup)
        self.current_state = self.maze.get_states()[position[0]][position[1]]
        self.start_position = position
        #now add initial state to the agents sample
        self.sample.append(self.current_state)

    def get_current_state(self):
        return self.current_state
    
    def get_sample(self):
        return self.sample
    
    def bellman(self, reward_sprime: int, utility_sprime: float, learning_rate: float, delta=0.01):
        return reward_sprime + learning_rate*utility_sprime

    def limit_actionspace_by_bounderies(self, actionspace: list, current_position: tuple):
        '''
        removes all unavailable actions from the actionspace.
        (position (3,0) has no business moving right in a 4x4 environment..)
        '''
        #if current position is on the left edge, remove left move
        if current_position[0]<=0:
            actionspace.remove((-1,0))
        #if current position is on the right edge, remove right move
        if current_position[0]>=self.maze.size[0]-1:
            actionspace.remove((1,0))
        #if current position is on the upper edge, remove up move
        if current_position[1]<=0:
            actionspace.remove((0,-1))
        #if current position is on the down edge, remove down move
        if current_position[1]>=self.maze.size[1]-1:
            actionspace.remove((0,1))
        return actionspace
    
    def run_through_maze(self,terminate_on_finalstate = False, limited_steps=1000):
        for i in range(limited_steps):
            self.act()
            if self.current_state.get_position() in self.maze.get_terminal_states() and terminate_on_finalstate:
                print("Agent has reached terminal state, Agent is now being reset")
                #reset state and position
                self.current_state=self.maze.get_states()[self.start_position[0]][self.start_position[1]]
                break
    
    def exploit(self,current_position: tuple, action: tuple) -> tuple:
        #makes the agent take an action - moving to another cell
        new_position = (current_position[0]+action[0],current_position[1]+action[1])
        print(f"old position: {current_position}, action: {action}, new position: {new_position}, ",end="")
        new_state = self.states[new_position[0]][new_position[1]]
        print(f"new state: {new_state}")
        #we only return the new state, all values (pos,reward) are just attribute to that state
        return new_state
    
    def explore(self,state,limited_actionspace,state_matrix):
        '''
        go to the next in line state in order to loop through update the utility of all states 
        note this function assumes agent starts at position 0,0 and loops from left to right, from up to down
        '''
        state_position = state.get_position()
        if (1,0) in limited_actionspace:
            #if right move is in actionspace, go right
            new_state_position=(state_position[0]+1,state_position[1])
        elif (0,1) in limited_actionspace:
            #if agent is on right edge, but can still go down, go to the left edge and one cell down
            new_state_position=(0,state_position[1]+1)        
        else:
            #if agent cant go right nor down, terminate (or go to first state again)
            new_state_position=(0,0)
   
        print(f"old position: {state_position}, new position: {new_state_position}")
        x,y = new_state_position
        new_state = state_matrix[x][y]
        print(new_state.get_position())
        return new_state



    def act(self):
        print("\n")
        print(f"Start Agents act() function..")
        #first define current position and current chosen action by the policy.
        position_current_state = self.current_state.get_position()
        #chooses an action based on the policy, that reads the state and actionspace
        #we use list(..) to copy the value in maze, instead of changing it
        total_actionspace = list(self.maze.get_actionspace())

        #now we limit the action space based on the agents position
        limited_actionspace = self.limit_actionspace_by_bounderies(total_actionspace, position_current_state)
        print(f"limited actionspace: ",limited_actionspace)

        action, max_value = self.policy.select_action(limited_actionspace, self.current_state, self.maze.get_states(), self.maze.get_values(), self.bellman) 
        
        #before we update value_matrix we update utility to 0 when calculating utility for terminal state
        if self.current_state.get_position() in self.maze.get_terminal_states():
            max_value=0

        # #determine next state based on the best action
        # self.current_state = self.exploit(current_position=position_current_state,action=action)

        #determine next in line state in order to explore and update utilities of all states
        self.current_state = self.explore(self.current_state, limited_actionspace, self.maze.get_states())

        #update utility matrix based on the calculated value of the old state (so update state 0 with utility of the best next state)
        self.maze.update_value_matrix(max_value,position_current_state)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)


