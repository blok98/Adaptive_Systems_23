from Maze import Maze
from State import State
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
            #calculate position of possible next state by adding the action (0,1) to the current state (2,3)
            nextstate_position = (current_position[0]+action[0],current_position[1]+action[1])
            #define indexes as i,j for shorter code
            i,j = nextstate_position
            nextstate_reward = state_matrix[i][j].get_reward()
            nextstate_value = value_matrix[i][j]
            nextstates_rewards.append(nextstate_reward)
        return nextstates_rewards

    def select_action(self,action_space: list, current_state: State, state_env: list, value_matrix: list, value_function):
        #for now chose a random action
        #but make sure to not go out of bounds
        #its defined as (n_column,n_row). so (2,3) is 3 right and 4 under

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
            # at last we update current state value with the bellman expectation equation
            currentstate_value = value_function(nextstate_reward, nextstate_value, 0.01)
            if currentstate_value>=max_value: 
                max_value=currentstate_value
                best_action=action

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
    
    def run_through_maze(self):
        for i in range(100):
            self.act()
            if self.current_state.get_position() in self.maze.get_terminal_states():
                print("Agent has reached terminal state, Agent is now being reset")
                #reset state and position
                self.current_state=self.maze.get_states()[self.start_position[0]][self.start_position[1]]
                break

    def act(self):
        #first define current position and current chosen action by the policy.
        position_current_state = self.current_state.get_position()
        #chooses an action based on the policy, that reads the state and actionspace
        #we use list(..) to copy the value in maze, instead of changing it
        total_actionspace = list(self.maze.get_actionspace())

        #now we limit the action space based on the agents position
        limited_actionspace = self.limit_actionspace_by_bounderies(total_actionspace, position_current_state)

        action, max_value = self.policy.select_action(limited_actionspace, self.current_state, self.maze.get_states(), self.maze.get_values(), self.bellman) 
        
        #calculate new position based on action
        self.current_state = self.maze.step(current_position=position_current_state,action=action)

        #update utility matrix based on the calculated value of the current
        self.maze.update_value_matrix(max_value,position_current_state)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)


