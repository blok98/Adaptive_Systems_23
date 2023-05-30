from Maze import Maze
from State import State
import random
from time import sleep
import random 

class Policy():
    def __init__(self) -> None:
        self.size=(0,0)
        self.policy_space = [[(0, 0)] * 4 for i in range(4)]
    
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
        print(f"new_policyspace: {self.policy_space}")
        
    def get_policyspace(self):
        """returns 2D list of all actions chosen by the policy

        Returns:
            list: 2D list of all actions
        """
        return self.policy_space
    
    def TD_value_function(self, current_utility, reward_sprime, utility_sprime, discount_factor, learning_rate):
        return current_utility + learning_rate*(reward_sprime + discount_factor*utility_sprime-current_utility)
    
    def TD_evaluate(self, nextstate_utilities, current_utility, actionspace):
        #define lowest utility a.p. to overwrite until highest utility is written down
        max_utility=-10**10
        best_action = (0,0)
        for i in range(len(nextstate_utilities)):
            possible_utility = nextstate_utilities[i]
            if possible_utility>max_utility:
                max_utility=possible_utility
                best_action=actionspace[i]
        return max_utility,best_action

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
        print(f"   policy function select_action started..")
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
        print(f"   best action of policy is {best_action} with value {max_value}")

        return best_action, max_value, old_value

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
    
    def get_policy(self) -> None:
        """returns the policy of the policy of the agent. So all possible actions of the policy of the agent.

        Returns:
            list: all possible actions in form of tuples f.e. (0,1)=right`
        """
        return self.policy.get_policyspace()

    def bellman(self, reward_sprime: int, utility_sprime: float, discount_factor: float) -> float:
        """calculates bellman equation in deterministic world

        Args:
            reward_sprime (int): reward of the next state
            utility_sprime (float): utility of the next state
            learning_rate (float): discount factor that determines relevance of future utilities
            delta (float, optional): _description_. Defaults to 0.01.

        Returns:
            float: _description_
        """
        return reward_sprime + discount_factor*utility_sprime

    def limit_actionspace_by_bounderies(self, actionspace: list, current_position: tuple) -> list:
        """        removes all unavailable actions from the actionspace.
        (position (3,0) has no business moving right in a 4x4 environment..)

        Args:
            actionspace (list): list of all possible actions the agent is capable of making
            current_position (tuple): current position of the agents state

        Returns:
            _type_: list of all possible actions the agent can make based on its position (without actions that lead to 'out of scene')
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
    
    def iterate(self,exploration_rate,terminate_on_finalstate = False, delta=0.01, limited_steps=1000) -> str:
        """iterates agents act function to simulate the mazes tables
        Return:
            returns string that determines if model is converged, based on condition 'max utility-diff < delta'
        """
        utility_imporv_list = []
        for i in range(limited_steps):
            # call act() function of agent
            old_v, new_v = self.act(exploration_rate)
            utility_imporv_list.append(new_v-old_v)
            if self.current_state.get_position() in self.maze.get_terminal_states() and terminate_on_finalstate:
                print("Agent has reached terminal state, Agent is now being reset")
                #reset state and position
                self.current_state=self.maze.get_states()[self.start_position[0]][self.start_position[1]]
                break
        if max(utility_imporv_list)<delta:
            return "Converged"
        else:
            return "Not Converged"
    
    def exploit(self, current_position: tuple, action: tuple, states: list) -> State:
        """step event: calculating new position in simulation when exploiting the policy. So it will always take the best action

        Args:
            current_position (tuple): current states position
            action (tuple): best action chosen by policy (for example one to the right (0,1))
            states (list): 2D list of all states

        Returns:
            State: new state based on taken action
        """
        #makes the agent take an action - moving to another cell
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

    def perceive(self,current_position: tuple, action: tuple) -> tuple:
        vs,qvalues,rs_prime,nextstate = self.maze.step(current_position, action)
        return vs,qvalues,rs_prime,nextstate

    def act(self, exploration_rate: float) -> tuple:
        """Filter actions, apply policy and update values and new state. Also maze.step is called to move the agent.

        Args:
            exploration_rate (float, optional): Chance of exploring instead of exploiting. Defaults to 1.0
        Return:
            tuple of best action and old utility and updated utility 
        """
        print("\n")
        print(f"Start Agents act() function..")
        #first define current position and current chosen action by the policy.
        position_current_state = self.current_state.get_position()  #(0,0)
        #chooses an action based on the policy, that reads the state and actionspace
        #we use list(..) to copy the value in maze, instead of changing it
        total_actionspace = list(self.maze.get_actionspace())   #[(1,0),(-1,0)...]

        action = random.choice(total_actionspace)

        #perceive values of situation Agent is in (remember TD means agent only knows about current state & rewards and values of current state and next state)
        utility, nextstate_utilities, nextstate_reward, nextstate = self.perceive(position_current_state, action) #5, [4,5,2,2], 7
        print(f"next state utilities: {nextstate_utilities}")

        #update new state. This makes the agent know its new position
        self.current_state=nextstate

        max_utility, best_action = self.policy.TD_evaluate(nextstate_utilities, utility,total_actionspace)
        print(f"max found utility for next state {max_utility}")
        print(f"best found action for next state {best_action}")

        #calculate new utility...
        new_utility = self.policy.TD_value_function(utility,nextstate_reward,max_utility,1,1)

        #before we update value_matrix we update utility to 0 when calculating utility for terminal state
        if self.current_state.get_position() in self.maze.get_terminal_states():
            max_utility=0
        
        #update utility matrix based on the calculated value of the old state (so update state 0 with utility of the best next state)
        self.maze.update_qvalue(new_utility, position_current_state, action)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)

        #lastly update policy space with chosen action on old position
        self.policy.update_policyspace(action, position_current_state)

        return utility, max_utility


