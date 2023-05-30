from Maze import Maze
from State import State
import random
from time import sleep
import random 

class Policy():
    def __init__(self) -> None:
        self.size=(4,4)
        self.policy_space = [[(0, 0)] * 4 for i in range(4)]
        self.action_space = [(1,0),(-1,0),(0,1),(0,-1)]
        self.qvalue_matrix = [[]]
        self.set_initial_qvalues()

    def set_initial_qvalues(self) -> None:
        """Sets initial q values (state-action pairs) for Maze on 0. It can be random but cause terminal has to be 0 its easier to make everything 0.
            It contains all utilities for each Q(s,a) per state. So there are size(actionspace) Q values per state, for self.size() amount of states.
        """
        self.qvalue_matrix = [[[0 for z in self.action_space] for i in range(self.size[0])] for j in range(self.size[1])]
    
    def set_size(self,size: tuple) -> None:
        """set size to prevent going out of bounds

        Args:
            size (tuple): the shape of the grid, used for taking action
        """
        self.size=size
    
    def update_policyspace(self,chosen_action: tuple,position: tuple) -> None:
        """updates policy space of the Policy (and Agent) by replacing current action (tuple) by chosen action on old position

        Args:
            chosen_action (tuple): chosen action by the agent, and the select_action function from Policy
            position (tuple): current position of the agent
        """
        self.policy_space[position[0]][position[1]]=chosen_action
        # print(f"new_policyspace: {self.policy_space}")

    def update_qvalue(self, qvalue, position, action):
        #first we define actionindex of best action so we can place new qvalue on right position
        action_index = self.action_space.index(action)
        # print(f"update qvalue {qvalue}. qvalue {self.qvalue_matrix[position[0]][position[1]][action_index]} is replaced with {qvalue}.")
        self.qvalue_matrix[position[0]][position[1]][action_index] = qvalue

    def get_policyspace(self):
        """returns 2D list of all actions chosen by the policy

        Returns:
            list: 2D list of all actions
        """
        return self.policy_space
    
    def get_qvalues(self):
        return self.qvalue_matrix
    
    def TD_value_function(self, current_utility, reward_sprime, utility_sprime, discount_factor, learning_rate):
        return current_utility + learning_rate*(reward_sprime + discount_factor*utility_sprime-current_utility)
    
    def Qlearning_choose_action(self,current_position,epsilon):
        i,j = current_position
        if random.random()>epsilon:
            print("random action chosen")
            action = random.choice(self.action_space)
            max_qvalue = self.qvalue_matrix[i][j][self.action_space.index(action)]
        else:
            print("--max action chosen--")
            max_qvalue = max((self.qvalue_matrix[i][j]))
            action = self.action_space[self.qvalue_matrix[i][j].index(max_qvalue)]
        print(f"utility {max_qvalue} out of all utilities: {self.qvalue_matrix[i][j]} has been chosen")
        return action, max_qvalue

    def select_action(self,action_space: list, current_state: State, state_env: list, value_matrix: list, value_function) -> tuple:
        """determine best utility and best action by applying bellman expectation equation

        Args:
            action_space (list): list of all possible actions, limited by position
            current_state (State): current state of the Agent
            state_env (list): 2D list of all states in the environment
            value_matrix (list): 2D list of all utilities
            value_function (_type_): Bellman expectation equation defined in Agent

        Returns:
            tuple: tuple of the best action and the maximum achievable utility and the old utlity
        """
        #for now chose a random action
        #but make sure to not go out of bounds
        #its defined as (n_column,n_row). so (2,3) is 3 right and 4 under
        # print(f"   policy function select_action started..")
        current_position = current_state.get_position()
        #we define the initial max_value as low as possible 
        max_value = -9999
        best_action = (0,0)
        #first we define old utility, to track delta and determine convergion
        i,j = current_position
        old_value = value_matrix[i][j]
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
            # at last we update current state value with the bellman expectation equation
            currentstate_value = value_function(nextstate_reward, nextstate_value, 1)
            if currentstate_value>=max_value: 
                max_value=currentstate_value
                best_action=action
        # print(f"   best action of policy is {best_action} with value {max_value}")

        return best_action, max_value, old_value

class Agent():
    def __init__(self, grid: Maze, policy: Policy) -> None:
        self.env=grid
        self.policy = policy
        self.current_state = None
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
    
    def get_policy(self) -> None:
        """returns the policy of the policy of the agent. So all possible actions of the policy of the agent.

        Returns:
            list: all possible actions in form of tuples f.e. (0,1)=right`
        """
        return self.policy.get_policyspace()

    def perceive(self,qvalue_matrix: list, current_position: tuple, action: tuple) -> tuple:
        rs_prime,nextstate = self.maze.step(qvalue_matrix, current_position, action)
        return rs_prime,nextstate

    def act(self, discount_factor: float, learning_rate: float, epsilon: float) -> tuple:
        """Filter actions, apply policy and update values and new state. Also maze.step is called to move the agent.

        Args:
            exploration_rate (float, optional): Chance of exploring instead of exploiting. Defaults to 1.0
        Return:
            tuple of best action and old utility and updated utility 
        """
        # print("\n")
        # print(f"Start Agents act() function..")
        #first define current position and current chosen action by the policy.
        position_current_state = self.current_state.get_position()  #(0,0)
        #chooses an action based on the policy, that reads the state and actionspace
        #we use list(..) to copy the value in maze, instead of changing it
        total_actionspace = list(self.maze.get_actionspace())   #[(1,0),(-1,0)...]
        
        # print(f"chose action {action} out of {total_actionspace}")
        action, utility = self.policy.Qlearning_choose_action(position_current_state, epsilon)

        #perceive values of situation Agent is in (remember TD means agent only knows about current state & rewards and values of current state and next state)
        nextstate_reward, nextstate = self.perceive(self.policy.get_qvalues(),position_current_state,action) #5, [4,5,2,2], 7

        #update new state. This makes the agent know its new position
        self.current_state=nextstate

        #we re-use policy choose action function with epsilon=0 to get the max qvalue for the next state
        _, nextstate_utility = self.policy.Qlearning_choose_action(nextstate.get_position(),epsilon=0)

        #calculate new utility...
        new_utility = self.policy.TD_value_function(utility,nextstate_reward,nextstate_utility,discount_factor,learning_rate)

        #before we update value_matrix we update utility to 0 when calculating utility for terminal state
        if self.current_state.get_position() in self.maze.get_terminal_states():
            new_utility=0
        #if final state is reached, also put utility on 0.
        if position_current_state in self.maze.get_terminal_states():
            new_utility=0
        
        #update utility matrix based on the calculated value of the old state (so update state 0 with utility of the best next state)
        self.policy.update_qvalue(new_utility, position_current_state, action)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)

        #lastly update policy space with chosen action on old position
        self.policy.update_policyspace(action, position_current_state)
        print(f"utility: {utility}, first_action: {action}, next reward: {nextstate_reward}, next utility: {nextstate_utility}, updated_utility: {new_utility}, next_position: {nextstate.get_position()}")

        return utility, new_utility