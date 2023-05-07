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

    def select_action(self, action_space, next_state_info, value_function):
        '''
        selects the best action based on percieve of agent and the bellman function
        args
            action_space: list of all possible actions  
            next_state_info: tuple of reward, utility for each possible next state [not that values are in same order of actionspace: so action 0 has reward,utility at index 0]
            value_function: the bellman expectation function defined in Agent
        return
            best_action: tuple of best action that agent will take 
            max_value: the best utility of the next state, reached by taking the best action
        '''
        #for now chose a random action
        #but make sure to not go out of bounds
        #its defined as (n_column,n_row). so (2,3) is 3 right and 4 under

        #we define the initial max_value as low as possible 
        max_value = -9999
        best_action = (0,0)
        for i in range(len(action_space)):
            # first we retrieve the reward of the next state ( r(s') )
            nextstate_reward = next_state_info[i][0]
            # then we retrieve the value of the next state ( v(s') )
            nextstate_value = next_state_info[i][1]
            # at last we update current state value with the bellman expectation equation
            currentstate_value = value_function(nextstate_reward, nextstate_value, 0.01)
            if currentstate_value>=max_value: 
                max_value=currentstate_value
                best_action=action_space[i]

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
    
    def run_through_maze(self,limited_steps=1000):
        for i in range(limited_steps):
            self.act()
            if self.current_state.get_position() in self.maze.get_terminal_states():
                print("Agent has reached terminal state, Agent is now being reset")
                #reset state and position
                self.current_state=self.maze.get_states()[self.start_position[0]][self.start_position[1]]
                break
    
    def perceive(self, limited_actionspace: list):
        '''
        Agents perceive function
        args:
            limited_actionspace: all possible actions based on current position
        return:
            current_state_info: tuple of reward, utility of the current state
            neighbour_info: list of a tuple containing reward, utility for each possible next state
        '''
        current_position = self.current_state.get_position()
        reward,utility = self.maze.get_state_info(position=current_position)
        current_state_info = reward,utility
        neighbour_info = []
        for action in limited_actionspace:
            #calculate position of possible next state by adding the action (0,1) to the current state (2,3)
            nextstate_position = (current_position[0]+action[0],current_position[1]+action[1])
            reward,utility = self.maze.get_state_info(position=nextstate_position)
            neighbour_info.append((reward,utility))
        
        return current_state_info, neighbour_info
    
    def update(self, position_current_state: tuple, action: tuple, max_value: float) -> None:
        '''
        updates agent and maze by calculating next position based on chosen action, and retrieving next state from maze
        args
            position_current_state: tuple of current position of the state 
            action: tuple of chosen action
            max_value: maximum utility and new calculated utility
        '''
        new_state = self.maze.step(position_current_state, action)
        self.current_state = new_state
        #update utility matrix based on the calculated value of the current
        self.maze.update_value_matrix(max_value, position_current_state)

    def act(self):
        #first define current position and current chosen action by the policy.
        position_current_state = self.current_state.get_position()
        #chooses an action based on the policy, that reads the state and actionspace
        #we use list(..) to copy the value in maze, instead of changing it
        total_actionspace = list(self.maze.get_actionspace())

        #now we limit the action space based on the agents position
        limited_actionspace = self.limit_actionspace_by_bounderies(total_actionspace, position_current_state)


        # perceive function: retrieve rewards and utilities from current state and from all states in a one step reach (markov)
        current_state_info, neighbour_info = self.perceive(limited_actionspace)

        action, max_value = self.policy.select_action(limited_actionspace, neighbour_info, self.bellman) 
        
        #updating agent by calcuating next position and retrieving next state
        self.update(position_current_state=position_current_state,action=action,max_value=max_value)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)


