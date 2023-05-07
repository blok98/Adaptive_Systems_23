from Maze import Maze
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
    
    def get_transition_probability(self, probability_matrix: list, old_position: tuple, new_position: tuple) -> float:
        """ the transition prob is located on x=current state (listed from left to right, up to down) and y=next state (listed equally)
        note that it doesnt matter what probability we chose in the matrix, given the lookup is consistently, because we use arbitrary values
        index = x-coord * 4 + y-coordinate. So (1,3) becomes index 7. Meaning the row index counts for 4 and column index counts for 1.

        Args:
            probability_matrix (list): 2D list of all state transition probabilities
            old_position (tuple): coordinates of current state
            new_position (tuple): coordinates of next state

        Returns:
            _type_: _description_
        """

        return probability_matrix[old_position[0]*4+old_position[1]][new_position[0]*4+new_position[1]]

    def select_action(self,action_space: list, current_state: State, state_env: list, value_matrix: list, probability_matrix: list, value_function) -> tuple:
        """determine best utility and best action by applying bellman expectation equation

        Args:
            action_space (list): list of all possible actions, limited by position
            current_state (State): current state of the Agent
            state_env (list): 2D list of all states in the environment
            value_matrix (list): 2D list of all utilities
            probability_matrix (list): 2D list of all state transition probabilities
            value_function (_type_): Bellman expectation equation defined in Agent

        Returns:
            tuple: tuple of the best action and the maximum achievable utility
        """
        print(f"   policy function select_action started..")
        current_position = current_state.get_position()
        #we define the initial max_value as low as possible 
        max_value = -9999
        best_action = (0,0)
        for action in action_space:
            #calculate position of possible next state by adding the action (0,1) to the current state (2,3)
            #note that current_position is (y,x) so nextstate = (y+action_y,x+action_x)
            nextstate_position = (current_position[0]+action[0],current_position[1]+action[1])
            #define indexes as i,j for shorter code
            i,j = nextstate_position
            # first we retrieve the reward of the next state ( r(s') )
            nextstate_reward = state_env[i][j].get_reward()
            # then we retrieve the value of the next state ( v(s') )
            nextstate_value = value_matrix[i][j]
            print(f"   check action {action}.. nexstatereward = {nextstate_reward}, nextstatevalue = {nextstate_value}, both on position {i,j}.")
            #before updating value we have to determine probability to go to state with current action

            transition_probability = self.get_transition_probability(probability_matrix,current_position,nextstate_position)
            # at last we update current state value with the bellman expectation equation
            currentstate_value = round(value_function(nextstate_reward, nextstate_value, 1, transition_probability),7)
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
    
    def set_current_state(self, position: tuple) -> None:
        """set current state based on given position of initial state (maze.states are indexed based on position, so we can do an easy lookup),
        and add initial state to the agents sample

        Args:
            position (tuple): coordinates of the new state
        """
        self.current_state = self.maze.get_states()[position[0]][position[1]]
        self.start_position = position
        #now add initial state to the agents sample
        self.sample.append(self.current_state)

    def get_current_state(self) -> State:
        """returns current state

        Returns:
            State: State object of current state
        """
        return self.current_state
    
    def get_sample(self) -> list:
        """Returns the sample of the visited states, usefull for backtracing agents behaviour

        Returns:
            list: list of all visited states as stateobjects
        """
        return self.sample
    
    def bellman(self, reward_sprime: int, utility_sprime: float, discount_factor: float, probability: float, delta=0.01) -> float:
        """calculates bellman equation in deterministic world

        Args:
            reward_sprime (int): reward of the next state
            utility_sprime (float): utility of the next state
            discount_factor (float): discount factor that determines relevance of future utilities
            probability (float): chance of going to the next state
            delta (float, optional): _description_. Defaults to 0.01.
        Returns:
            float: _description_
        """
        return probability * (reward_sprime + discount_factor*utility_sprime)

    def limit_actionspace_by_bounderies(self, actionspace: list, current_position: tuple) -> list:
        """        removes all unavailable actions from the actionspace.
        (position (3,0) has no business moving right in a 4x4 environment..)

        Args:
            actionspace (list): list of all possible actions the agent is capable of making
            current_position (tuple): current position of the agents state

        Returns:
            list: list of all possible actions the agent can make based on its position (without actions that lead to 'out of scene')
        """
        #if current position is on the upper edge, remove upper move
        if current_position[0]<=0:
            actionspace.remove((-1,0))
        #if current position is on the lower edge, remove upper move
        if current_position[0]>=self.maze.size[0]-1:
            actionspace.remove((1,0))
        #if current position is on the left edge, remove left move
        if current_position[1]<=0:
            actionspace.remove((0,-1))
        #if current position is on the right edge, remove right move
        if current_position[1]>=self.maze.size[1]-1:
            actionspace.remove((0,1))
        return actionspace
    
    def iterate(self,exploration_rate,terminate_on_finalstate = False, limited_steps=1000) -> None:
        """iterates agents act function to simulate the mazes tables
        """
        for i in range(limited_steps):
            self.act(exploration_rate)
            if self.current_state.get_position() in self.maze.get_terminal_states() and terminate_on_finalstate:
                print("Agent has reached terminal state, Agent is now being reset")
                #reset state and position
                self.current_state=self.maze.get_states()[self.start_position[0]][self.start_position[1]]
                break
    
    def exploit(self, current_position: tuple, action: tuple, states: list) -> State:
        """step event: calculating new position in simulation when exploiting the policy. So it will always take the best action

        Args:
            current_position (tuple): current states position
            action (tuple): best action chosen by policy (for example one to the right (0,1))
            states (list): 2D list of all states

        Returns:
            State: new state based on taken action
        """
        new_position = (current_position[0]+action[0],current_position[1]+action[1])
        print(f"old position: {current_position}, action: {action}, new position: {new_position}, ",end="")
        new_state = states[new_position[0]][new_position[1]]
        print(f"new state: {new_state}")
        #we only return the new state, all values (pos,reward) are just attribute to that state
        return new_state
    
    def explore(self,state: State,limited_actionspace: list,state_matrix: list) -> State:
        """ go to the next in line state in order to loop through update the utility of all states 
        note this function assumes agent starts at position 0,0 and loops from left to right, from up to down

        Args:
            state (State): current state
            limited_actionspace (list): all possible actions based on current position
            state_matrix (list): 2D list of all states

        Returns:
            State: new state based on taken action
        """
        state_position = state.get_position()
        if (0,1) in limited_actionspace:
            #if right move is in actionspace, go right
            new_state_position=(state_position[0],state_position[1]+1)
        elif (1,0) in limited_actionspace:
            #if agent is on right edge, but can still go down, go to the left edge and one cell down
            new_state_position=(state_position[0]+1,0)        
        else:
            #if agent cant go right nor down, terminate (or go to first state again)
            new_state_position=(0,0)
   
        print(f"old position: {state_position}, new position: {new_state_position}")
        x,y = new_state_position
        new_state = state_matrix[x][y]
        print(new_state.get_position())
        return new_state



    def act(self, exploration_rate: float = 1.0) -> None:
        """Filter actions, apply policy and update values and new state. Also maze.step is called to move the agent.
        
        Args:
            exploration_rate (float, optional): Chance of exploring instead of exploiting. Defaults to 1.0.
        """
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

        action, max_value = self.policy.select_action(limited_actionspace, self.current_state, self.maze.get_states(), self.maze.get_values(), self.maze.get_probability_matrix(), self.bellman) 

        #before we update value_matrix we update utility to 0 when calculating utility for terminal state
        if self.current_state.get_position() in self.maze.get_terminal_states():
            max_value=0
        
        #checks wether to exploit or explore
        exploit_status = random.random()>exploration_rate
        if exploit_status:
            # #determine next state based on the best action
            self.current_state = self.exploit(position_current_state,action, self.maze.get_states())
        else:
            #determine next in line state in order to explore and update utilities of all states
            self.current_state = self.explore(self.current_state, limited_actionspace, self.maze.get_states())

        #update utility matrix based on the calculated value of the old state (so update state 0 with utility of the best next state)
        self.maze.update_value_matrix(max_value,position_current_state)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)


