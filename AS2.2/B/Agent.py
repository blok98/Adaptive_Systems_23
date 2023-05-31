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
        self.utility_matrix = [[]]
        self.set_initial_utilities()
    
    def set_size(self,size: tuple) -> None:
        """set size to prevent going out of bounds

        Args:
            size (tuple): the shape of the grid, used for taking action
        """
        self.size=size

    def set_initial_utilities(self) -> None:
        """Sets initial q values (state-action pairs) for Maze on 0. It can be random but cause terminal has to be 0 its easier to make everything 0.
            It contains all utilities for each Q(s,a) per state. So there are size(actionspace) Q values per state, for self.size() amount of states.
        """
        self.utility_matrix = [[0 for i in range(self.size[0])] for j in range(self.size[1])]
    
    def update_policyspace(self,chosen_action: tuple,position: tuple) -> None:
        """updates policy space of the Policy (and Agent) by replacing current action (tuple) by chosen action on old position

        Args:
            chosen_action (tuple): chosen action by the agent, and the select_action function from Policy
            position (tuple): current position of the agent
        """
        self.policy_space[position[0]][position[1]]=chosen_action
        # print(f"new_policyspace: {self.policy_space}")

    def update_utility(self, utility: float, position: tuple) -> None:
        """Updates new calculated q value by replacing the corresponding value in the qvalue matrix

        Args:
            utility (float): new calculated utility: V(s)
            position (tuple): current state: s
        """
        i,j = position
        self.utility_matrix[i][j] = utility

    def get_policyspace(self) -> list:
        """returns 2D list of all actions chosen by the policy

        Returns:
            list: 2D list of all actions
        """
        return self.policy_space
    
    def get_utilities(self) -> list:
        return self.utility_matrix
    
    def get_utility(self,position: tuple) -> float:
        """Returns a utility on a given position, is used to determine v(s') given new position

        Args:
            position (tuple): position of next state

        Returns:
            float: utility of next state
        """
        i,j = position
        return self.utility_matrix[i][j]
    
    def TD_value_function(self, current_utility: float, reward_sprime: float, utility_sprime: float, discount_factor: float, learning_rate: float) -> float:
        """calculates new qvalue based on the TD-formula

        Args:
            current_utility (float): qvalue from current position to next position: Q(s,a)
            reward_sprime (float): reward R of next state s'
            utility_sprime (float): max qvalue from next state to the state after that: Q(s',a')
            discount_factor (float): weighting of future expected returns: Y
            learning_rate (float): weighting of error: a

        Returns:
            float: new calculated qvalue of current state to next position: Q(s,a)
        """
        return current_utility + learning_rate*(reward_sprime + discount_factor*utility_sprime-current_utility)
    
    def TD_choose_action(self,current_position: tuple, epsilon:float) -> tuple:
        """chooses action and qvalue randomly or according to ARGMAX depending on epsilon.
        This function is called twice for the first step (Q(s,a)) and the second step (Q(s',a'))
        Second step has no exploration so at the second step epsilon is set to 0

        Args:
            current_position (tuple): current position of the agent: s
            epsilon (float): chance to explore instead of exploit: e

        Returns:
            tuple: chosen action 
        """
        i,j = current_position
        #TD-learning is evaluation on policy, so we determine action based on policy
        action = self.policy_space[i][j]
        return action

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

    def perceive(self,utility_matrix: list, current_position: tuple, action: tuple) -> tuple:
        """Perceive function of agent that calls the Maze.step function and observes reward and new position

        Args:
            qvalue_matrix (list): all qvalues known to the agent, located in Policy
            current_position (tuple): current position of the agent: s
            action (tuple): action chosen by TD_choose_action(): a

        Returns:
            tuple: reward of next state, State object of next state
        """
        rs_prime,nextstate = self.maze.step(utility_matrix, current_position, action)
        return rs_prime,nextstate

    def act(self, discount_factor: float, learning_rate: float, epsilon: float) -> tuple:
        """Filter actions, apply policy and update values and new state. Also maze.step is called to move the agent.

        Args:
            discount_factor (float): weighting of future expected returns: Y
            learning_rate (float): weighting of error: a
            epsilon (float): chance of exploring instead of exploiting

        Returns:
            tuple: returns the old qvalue and the new calculated qvalue
        """
        position_current_state = self.current_state.get_position()  #(0,0)
        
        action = self.policy.TD_choose_action(position_current_state, epsilon)

        #perceive values of situation Agent is in (remember TD means agent only knows about current state & rewards and values of current state and next state)
        nextstate_reward, nextstate = self.perceive(self.policy.get_utilities(),position_current_state,action) #5, [4,5,2,2], 7
        print(position_current_state,nextstate.get_position())
        #update new state. This makes the agent know its new position
        self.current_state=nextstate

        #ask for utility at current state and next state from policy (V(s) & V(s'))
        utility = self.policy.get_utility(position_current_state)
        nextstate_utility = self.policy.get_utility(nextstate.get_position())

        #calculate new utility (V(s))...
        new_utility = self.policy.TD_value_function(utility,nextstate_reward,nextstate_utility,discount_factor,learning_rate)

        #if final state is reached, also put utility on 0. We look at old position, not the updated one cause we want to determine place of current state.
        if position_current_state in self.maze.get_terminal_states():
            new_utility=0
        
        #update utility matrix based on the calculated value of the old state (so update state 0 with utility of the best next state)
        self.policy.update_utility(new_utility, position_current_state)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)

        #lastly update policy space with chosen action on old position
        self.policy.update_policyspace(action, position_current_state)
        #print(f"utility: {utility}, first_action: {action}, next reward: {nextstate_reward}, next utility: {nextstate_utility}, updated_utility: {new_utility}, next_position: {nextstate.get_position()}")

        return utility, new_utility