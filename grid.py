import pygame
import random
import math
# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 600, 600
LINE_WIDTH = 10
BOARD_ROWS, BOARD_COLS = 3, 3
CELL_SIZE = WIDTH // BOARD_COLS
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
grid=[['' for _ in range(3)]for _ in range(3)]
# Set up display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Tic Tac Toe")
screen.fill(WHITE)
count=0
position=[(i,j)for i in range(3)for j in range(3)]
empty_squares=9
def draw():
    return all(grid[i][j] != '' for i in range(3) for j in range(3)) or empty_squares==0
def win():
    for i in range(3):
        if(grid[i][0]==grid[i][1] and grid[i][2]==grid[i][0]and grid[i][0]!=''):
            return True,grid[i][0]
        if(grid[0][i]==grid[1][i] and grid[0][i]==grid[2][i] and grid[0][i]!=''):
            return True,grid[0][i]
    return (grid[0][2]==grid[1][1]and grid[2][0]==grid[1][1] and grid[1][1]!='')or(grid[1][1]==grid[0][0] and grid[2][2]==grid[1][1] and grid[1][1]!=''),grid[1][1]
def pc():
    x = {0: CELL_SIZE // 2 - 40, 1: CELL_SIZE+CELL_SIZE // 2 - 40 , 2: 2 * CELL_SIZE + CELL_SIZE // 2 - 40}
    y = {0: CELL_SIZE // 2 - 40, 1: CELL_SIZE+CELL_SIZE // 2 - 40, 2: 2 * CELL_SIZE + CELL_SIZE // 2 - 40}
    font = pygame.font.SysFont('Corbel', 100)
    O = font.render('O', True, BLACK)

    if count == 1:
        place = random.randint(0,8)
        while(grid[place//3][place%3]=='x'):
            place=random.randint(0,8)
        x_place,y_place=place//3,place%3
        grid[x_place][y_place]='o'
        print(place)
        position.remove((x_place,y_place))
        print(grid)
    else:
        if win()[0]:
            return
        elif draw():
            font = pygame.font.SysFont('calisto', 50)
            d = font.render("DRAW", True,BLACK)
            return
        move = minimax(empty_squares,True)  
        x_place, y_place = move['position']
        grid[x_place][y_place] = 'o'
        position.remove((move['position']))
    print(position)
    screen.blit(O, (y[y_place],x[x_place]))
    pygame.display.update()

def minimax(empty_places, is_maximizing):
    result = win()
    if result[0]:
        return {'position': None, 'score': empty_places if result[1] == 'o' else -empty_places}
    if draw():
        return {'position': None, 'score': 0}

    best = {'position': None, 'score': -math.inf if is_maximizing else math.inf}
    local_positions = position[:]

    for i, j in local_positions:
        grid[i][j] = 'o' if is_maximizing else 'x'
        position.remove((i, j))
        score = minimax(empty_places - 1, not is_maximizing)
        score['position'] = (i, j)
        grid[i][j] = ''
        position.append((i, j))
        if (is_maximizing and score['score'] > best['score']) or (not is_maximizing and score['score'] < best['score']):
            best = score
    return best


def draw_grid():
    for row in range(1, BOARD_ROWS):
        pygame.draw.line(screen, BLACK, (0, row * CELL_SIZE), (WIDTH, row * CELL_SIZE), LINE_WIDTH)
    for col in range(1, BOARD_COLS):
        pygame.draw.line(screen, BLACK, (col * CELL_SIZE, 0), (col * CELL_SIZE, HEIGHT), LINE_WIDTH)

draw_grid()
pygame.display.update()
font=pygame.font.SysFont('Corbel',100)
X=font.render('X',True,BLACK)
# Main loop
running = True
end=False
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        if event.type==pygame.MOUSEBUTTONDOWN:
            if mouse[0]<CELL_SIZE and mouse[1]<CELL_SIZE and grid[0][0]=='' and not end:
                screen.blit(X,(CELL_SIZE//2-40,CELL_SIZE//2-40))
                grid[0][0]='x'
                position.remove((0,0))
            elif mouse[0]<CELL_SIZE*2 and mouse[1]<CELL_SIZE and grid[0][1]=='' and not end:
                screen.blit(X,(CELL_SIZE+CELL_SIZE//2-40,CELL_SIZE//2-40))
                grid[0][1]='x'
                position.remove((0,1))
            elif mouse[0]<CELL_SIZE*3 and mouse[1]<CELL_SIZE and grid[0][2]=='' and not end:
                screen.blit(X,(2*CELL_SIZE+CELL_SIZE//2-40,CELL_SIZE//2-40))
                grid[0][2]='x'
                position.remove((0,2))
            elif mouse[0]<CELL_SIZE and mouse[1]<CELL_SIZE*2 and grid[1][0]=='' and not end:
                screen.blit(X,(CELL_SIZE//2-40,CELL_SIZE+CELL_SIZE//2-40))
                grid[1][0]='x'
                position.remove((1,0))
            elif mouse[0]<CELL_SIZE*2 and mouse[1]<CELL_SIZE*2 and grid[1][1]=='' and not end:
                screen.blit(X,(CELL_SIZE+CELL_SIZE//2-40,CELL_SIZE+CELL_SIZE//2-40))
                grid[1][1]='x'
                position.remove((1,1))
            elif mouse[0]<CELL_SIZE*3 and mouse[1]<CELL_SIZE*2 and grid[1][2]=='' and not end:
                screen.blit(X,(2*CELL_SIZE+CELL_SIZE//2-40,CELL_SIZE+CELL_SIZE//2-40))
                grid[1][2]='x'
                position.remove((1,2))
            elif mouse[0]<CELL_SIZE and mouse[1]<CELL_SIZE*3 and grid[2][0]=='' and not end:
                screen.blit(X,(CELL_SIZE//2-40,2*CELL_SIZE+CELL_SIZE//2-40))
                grid[2][0]='x'
                position.remove((2,0))
            elif mouse[0]<CELL_SIZE*2 and mouse[1]<CELL_SIZE*3 and grid[2][1]=='' and not end:
                screen.blit(X,(CELL_SIZE+CELL_SIZE//2-40,2*CELL_SIZE+CELL_SIZE//2-40))
                grid[2][1]='x'
                position.remove((2,1))
            elif mouse[0]<CELL_SIZE*3 and mouse[1]<CELL_SIZE*3 and grid[2][2]=='' and not end:
                screen.blit(X,(CELL_SIZE*2+CELL_SIZE//2-40,2*CELL_SIZE+CELL_SIZE//2-40))                
                grid[2][2]='x'
                position.remove((2,2))
            else:
                count-=1
                empty_squares+=1
            count+=1
            empty_squares-=1
            print(position)
            pc()
    mouse = pygame.mouse.get_pos()
    pygame.display.update()
    if win()[0]:
        end=True
        winner_font=pygame.font.SysFont('microsoftsansserif',50)
        if win()[1]=='x':
            winner=winner_font.render('You are the winner',True,(255,204,0))
        else:
            winner=winner_font.render("the pc is the winner",True,(255,204,0))
        screen.blit(winner,(100,200))    
    elif draw():
        font=pygame.font.SysFont('calisto',50)
        d=font.render("DRAW",True,BLACK)
        screen.blit(d,(200,200))
    pygame.display.update()
pygame.quit()
