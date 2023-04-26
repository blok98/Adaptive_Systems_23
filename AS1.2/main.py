from Maze import Maze
from Policy import Policy
from Agent import Agent
from State import State

#its defined as (n_column,n_row). so (2,3) is 3 right and 4 under"

if __name__=="__main__":
    reward_matrix = [[-1,-1,-1,40],
                     [-1,-1,-10,-10],
                     [-1,-1,-1,-1],
                     [10,-2,-1,-1]]
    #final states (finish) are located in the upper right and lower left corners
    final_states = [(3,0),(0,3)]
    starting_position = (2,3)
    m1 = Maze(reward_matrix, final_states)
    p1 = Policy()
    a1 = Agent(m1,p1)
    a1.set_current_state(starting_position)
    pos = a1.get_current_state().get_position()

    for i in range(100):
        a1.act()
        if a1.get_current_state().get_position() in final_states:
            break
    