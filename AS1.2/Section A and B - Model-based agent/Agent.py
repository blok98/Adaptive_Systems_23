from Maze import Maze
from State import State
from Policy import Policy


class Agent():
    def __init__(self, grid: Maze, policy: Policy) -> None:
        self.env=grid
        self.policy = policy
        self.current_state = None
        self.maze = grid
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

    def limit_actionspace_by_bounderies(self, actionspace: list, current_position: tuple) -> list:
        """        removes all unavailable actions from the actionspace.
        (position (3,0) has no business moving right in a 4x4 environment..)

        Args:
            actionspace (list): list of all possible actions the agent is capable of making
            current_position (tuple): current position of the agents state

        Returns:
            _type_: list of all possible actions the agent can make based on its position (without actions that lead to 'out of scene')
        """
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
    
    def run_through_maze(self) -> None:
        """iterates agents act function to simulate the mazes tables
        """
        for i in range(100):
            self.act()
            if self.current_state.get_position() in self.maze.get_terminal_states():
                break

    def act(self) -> None:
        """Filter actions, apply policy and update values and new state. Also maze.step is called to move the agent.
        """
        #first define current position and current chosen action by the policy.
        position = self.current_state.get_position()
        #chooses an action based on the policy, that reads the state and actionspace
        #we use list(..) to copy the value in maze, instead of changing it
        total_actionspace = list(self.maze.get_actionspace())

        #now we limit the action space based on the agents position
        limited_actionspace = self.limit_actionspace_by_bounderies(total_actionspace, position)

        action = self.policy.select_action(limited_actionspace) 
        
        #then calculate new position based on action
        self.current_state = self.maze.step(current_position=position,action=action)

        #now add new visited state to the agents sample
        self.sample.append(self.current_state)
