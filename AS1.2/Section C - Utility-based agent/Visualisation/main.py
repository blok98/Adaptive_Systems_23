from Maze import Maze
from Agent import Agent,Policy
from State import State
import pygame
from time import sleep
#its defined as (n_row,n_column). so (2,3) is 3 under and 4 right"

if __name__=="__main__":
    reward_matrix = [[-1,-1,-1,40],
                     [-1,-1,-10,-10],
                     [-1,-1,-1,-1],
                     [10,-2,-1,-1]]
    #final states (finish) are located in the upper right and lower left corners
    final_states = [(3,0),(0,3)]
    #we chose starting point (0,0) because we want to explore the env and loop through all states
    starting_position = (0,0)
    m1 = Maze(reward_matrix, final_states)
    p1 = Policy()
    a1 = Agent(m1,p1)
    a1.set_current_state(starting_position)



    #define colors we gonne use for the visualisation cells
    WHITE = (255, 255, 255)
    GRAY = (200, 200, 200)
    GREEN = (0, 255, 0)
    RED = (255, 0, 0)
    YELLOW = (255, 200, 0)
    ORANGE = (255, 120, 0)
    #to highlight the agents position
    BLUE = (37, 150, 190)
    LIGHTBLUE = (205, 240, 255)

    pygame.init()

    #set the window size based on the maze size, so the scale=cell size
    scale = 100
    shape=m1.get_size()
    #use scale to multiply the maze size
    WINDOW_SIZE = (shape[0]*scale,shape[1]*scale)
    screen = pygame.display.set_mode(WINDOW_SIZE)

    #define the grid based on the predefined shape and scale
    GRID_WIDTH = shape[0]
    GRID_HEIGHT = shape[1]
    CELL_SIZE = scale
    # set the structure of the grid based on the mazes environment
    grid = m1.get_values()

    #sets the font for values
    font = pygame.font.Font(None, 22)

    #set up the button that switches between utility and reward visualisations
    button_rect = pygame.Rect(110, 0, 150, 25)
    button_color = pygame.Color('white')
    button_text = font.render('press:', True, pygame.Color('black'))

    #sets the font for values
    font = pygame.font.Font(None, 30)

    #set the utility values of the maze as the main values for the cells. 
    status_visualisation = 0
    grid = m1.get_values()
    policy_space = a1.policy.get_policyspace()

    #here we define the settings and condition for stopping when model is converged
    utility_imporv_list = []
    delta = 0.01

    #first we make a game loop in which we update the visualisation based on the agents actions
    running = True
    status = "Not converged"
    while running:

        #define current position
        current_position=a1.get_current_state().get_position()
        #make sure that when model is converged agent stays at position 3,3 so the model stops running (it doenst but i wont explore the world anymore)
        if status == "Converged":
            current_position=(3,3)
            
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                #check if the button was clicked and if so, change visualised numbers to rewards or utilities
                if button_rect.collidepoint(event.pos):
                    # now we check current visualisation (status_visualisation) and show the other
                    if status_visualisation==0:
                        grid = m1.get_reward_matrix()
                        status_visualisation=1
                    elif status_visualisation==1:
                        grid = m1.get_values()
                        status_visualisation=0

        screen.fill(WHITE)

        #draw the grid. 
        #We switch row and column for drawing the states, because we define coordinates as (x,y).
        #So a write for x=2, y=3 is done by writing to grid[2][3] or grid[column][row]
        for row in range(GRID_HEIGHT):
            for column in range(GRID_WIDTH):
                x = column * CELL_SIZE
                y = row * CELL_SIZE

                #determine the color of the cell based on the utility value of the cell

                if grid[row][column] > 1:
                    color = GREEN
                elif grid[row][column]  >= 0:
                    color = YELLOW
                elif grid[row][column]  >= -1:
                    color = ORANGE
                else:
                    color = RED

                #now we recolor the cell the agent is currently in
                #we make sure to flip the coordinates (column,row) because we maintained (x,y) coordinates (probably stupid)
                if (row,column) == current_position:
                    color = BLUE
                #now recolor start position
                if (row,column) == starting_position:
                    color = LIGHTBLUE

                #draw the cell in the window with corresponding cell, and color info
                pygame.draw.rect(screen, color, [x, y, CELL_SIZE, CELL_SIZE])

                #draw the utility value in the cell
                text = font.render(str(grid[row][column]), True, (0, 0, 0))
                text_rect = text.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 2))
                screen.blit(text, text_rect)

                #draw the action in tuples in the cell
                #first draw small font size
                small_font = pygame.font.Font(None, 12)
                text = small_font.render(str(policy_space[row][column]), True, (0, 0, 0))
                text_rect = text.get_rect(center=(x + CELL_SIZE / 2, y + CELL_SIZE / 1.5))
                screen.blit(text, text_rect)

                
                pygame.draw.rect(screen, button_color, button_rect)
                current_visualisation=["utility","reward"][status_visualisation]   
                button_text=f"{button_text} {current_visualisation}"
                button_text = font.render(f'press: {current_visualisation}', True, pygame.Color('black'))
                screen.blit(button_text,(110,0))
        


        #update the display
        pygame.display.update()
        
        #move one step with agent
        old_v, new_v = a1.act(exploration_rate=1)
        utility_imporv_list.append(new_v-old_v)
        
        #check for convergion to terminate or not
        if current_position == (3,3):
            if max(utility_imporv_list)<delta:
                status = "Converged"
            else:
                status =  "Not Converged"

            utility_imporv_list=[]

        #set timer for better visualisation
        sleep(0.2)

    #quit pygame
    pygame.quit()

    sample = a1.get_sample()

    # print("sample: ", [state.get_position() for state in sample])
    print("\n\nModel has converged..")
    print("value matrix: ", m1.value_matrix)
    print("policy: ",a1.get_policy())