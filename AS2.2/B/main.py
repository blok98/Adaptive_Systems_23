from Maze import Maze
from Agent import Agent,Policy
from State import State
import pygame
from time import sleep
#its defined as (n_row,n_column). so (2,3) is 3 under and 4 right"

def run(a1: Agent,m1: Maze, step_time: int, discount_factor=1,learning_rate=1,epsilon=0.2,n_episodes=1, endvisualisation_time=50):
    # its defined as (n_row,n_column). so (2,3) is 3 under and 4 right"
    status = "Not converged"
    episode_num=0
    if status=="Not converged":
        for i in range(999999999):
            if episode_num>=n_episodes:
                break
            # move one step with agent
            old_v, new_v = a1.act(discount_factor,learning_rate,epsilon)
            current_position=a1.get_current_state().get_position()
            if current_position in m1.get_terminal_states():
                status="Converged"
                a1.set_current_state((3,2))
                episode_num+=1
                status="Not Converged"
    print(f"episode {episode_num} has converged")
    visualize_maze(a1,m1,step_time,discount_factor,learning_rate,epsilon,endvisualisation_time)

def visualize_maze(a1: Agent,m1: Maze, step_time: int, discount_factor=1,learning_rate=1,epsilon=0.2,n_episodes=1):
    # define colors we gonne use for the visualisation cells
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 200, 0)
    ORANGE = (255, 120, 0)
    # to highlight the agents position
    BLUE = (37, 150, 190)
    LIGHTBLUE = (205, 240, 255)

    pygame.init()

    # set the window size based on the maze size, so the scale=cell size
    scale = 100
    shape=m1.get_size()
    # use scale to multiply the maze size
    WINDOW_SIZE = (shape[0]*scale,shape[1]*scale)
    screen = pygame.display.set_mode(WINDOW_SIZE)

    # define the grid based on the predefined shape and scale
    GRID_WIDTH = shape[0]
    GRID_HEIGHT = shape[1]
    CELL_SIZE = scale
    # set the structure of the grid based on the mazes environment
    grid = p1.utility_matrix

    # sets the font for values
    font = pygame.font.Font(None, 22)

    # set up the button that switches between utility and reward visualisations
    button_rect = pygame.Rect(110, 0, 150, 25)
    button_color = pygame.Color('white')
    button_text = font.render('press:', True, pygame.Color('black'))

    # sets the font for values
    font = pygame.font.Font(None, 30)

    # set the utility values of the maze as the main values for the cells.
    status_visualisation = 0
    grid = p1.get_utilities()
    policy_space = a1.policy.get_policyspace()

    # first we make a game loop in which we update the visualisation based on the agents actions
    running = True
    status = "Not converged"
    episode_num = 0
    while running:
        # define current position
        current_position=a1.get_current_state().get_position()
        # make sure that when model is converged agent stays at position 3,3 so the model stops running (it doesn't but i wont explore the world anymore)
        if current_position in m1.get_terminal_states():
            status="Converged"
        
        if status=="Converged":
            a1.set_current_state((3,2))
            print(f"episode {episode_num} has converged")
            episode_num+=1
            status="Not Converged"
        
        if episode_num>=n_episodes:
            break

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                # check if the button was clicked and if so, change visualised numbers to rewards or utilities
                if button_rect.collidepoint(event.pos):
                    # now we check current visualisation (status_visualisation) and show the other
                    if status_visualisation==0:
                        grid = m1.get_reward_matrix()
                        status_visualisation=1
                    elif status_visualisation==1:
                        grid = p1.get_utilities()
                        status_visualisation=0

        screen.fill(WHITE)

        # draw the grid.
        for row in range(GRID_HEIGHT):
            for column in range(GRID_WIDTH):
                x = column * CELL_SIZE
                y = row * CELL_SIZE

                # determine the color of the cell based on the utility value of the cell
                if grid[row][column] > 1:
                    color = GREEN
                elif grid[row][column]  >= 0:
                    color = YELLOW
                elif grid[row][column]  >= -1:
                    color = ORANGE
                else:
                    color = RED

                # now we recolor the cell the agent is currently in
                if (row,column) == current_position:
                    color = BLUE
                # now recolor start position
                if (row,column) == starting_position:
                    color = LIGHTBLUE

                # draw the cell in the window with corresponding cell, and color info
                pygame.draw.rect(screen, color, [x, y, CELL_SIZE, CELL_SIZE])

                font = pygame.font.Font(None, 20)
                # draw the utility value in the cell
                # Definieer een lijst met relatieve posities voor elke richting (boven, onder, links, rechts)
                directions = [(0, 1), (0, -1), (1, 0), (-1, 0),(0,0)]  # [rechts, links, omlaag, omhoog]
                direction_labels = ['>', '<', 'v', '^','']

                # Tekenen van de waarde op de juiste positie in de cel
                action_value = grid[row][column]
                text = font.render(str(round(action_value,2)), True, (0, 0, 0))
                text_rect = text.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 2))
                screen.blit(text, text_rect)



                # draw the action in tuples in the cell
                # first draw small font size
                small_font = pygame.font.Font(None, 40)
                text = small_font.render(str(direction_labels[directions.index(policy_space[row][column])]), True, (0, 0, 0))
                text_rect = text.get_rect(center=(x + CELL_SIZE / 1.5, y + CELL_SIZE / 1.5))
                screen.blit(text, text_rect)

        pygame.draw.rect(screen, button_color, button_rect)
        current_visualisation=["utility","reward"][status_visualisation]   
        button_text=f"{button_text} {current_visualisation}"
        button_text = font.render(f'press: {current_visualisation}', True, pygame.Color('black'))
        screen.blit(button_text,(110,0))

        # update the display
        pygame.display.update()

        # move one step with agent
        old_v, new_v = a1.act(discount_factor,learning_rate,epsilon)

        # set timer for better visualization
        sleep(step_time)

    # quit pygame
    pygame.quit()

if __name__ == "__main__":
    reward_matrix = [[-1,-1,-1,40],
                    [-1,-1,-10,-10],
                    [-1,-1,-1,-1],
                    [10,-2,-1,-1]]
    #we define the policy determined by the earlier assignment "1.2 utility based agent"
    policyspace = [[(0, 1), (0, 1), (0, 1), (0, -1)], 
     [(0, 1), (-1, 0), (-1, 0), (-1, 0)], 
     [(0, 1), (-1, 0), (0, -1), (0, -1)], 
     [(-1, 0), (-1, 0), (-1, 0), (0, -1)]]
    
    # final states (finish) are located in the upper right and lower left corners
    final_states = [(3,0),(0,3)]
    #we now define start position on the official start position (3,4)
    starting_position = (3,2)
    m1 = Maze(reward_matrix, final_states)
    p1 = Policy()
    p1.set_policyspace(policyspace)
    a1 = Agent(m1,p1)
    a1.set_current_state(starting_position)

    policy = a1.get_policy()
    run(a1,m1, step_time = 0.3, discount_factor=1,learning_rate=0.2,epsilon=0.2,n_episodes=10000)
    #visualize_maze(a1,m1,step_time = 0.3,discount_factor=1,learning_rate=0.2,epsilon=0.2, n_episodes=100)
    print(f'\n\n new qvalue matrix: {p1.utility_matrix}')
