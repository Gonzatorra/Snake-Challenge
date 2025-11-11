#----------------------------------------------------#
#-----------------------Step 1-----------------------#
#-----Play this to collect human data by playing-----#
#----------------------------------------------------#



#------------Import Libraries------------#
import pickle
import pygame
import os
import sys




#------------Set up directories------------#
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "snake_env"))
from env import SnakeEnv





#------------Set up basic------------#
pygame.init()
env = SnakeEnv(render_mode=None)  # deshabilitamos cv2
obs, _ = env.reset()
done = False
WINDOW_SIZE = 500
cell_size = 10
screen = pygame.display.set_mode((WINDOW_SIZE, WINDOW_SIZE))
pygame.display.set_caption("Snake Game - Human Play")
clock = pygame.time.Clock()

#Start with one action
action = 1
human_data = []
print("Controls: Arrows LEFT/RIGHT/UP/DOWN. Press ESC to end the episode.")





#------------Function for show game screen------------#
def draw_env(env, screen):
    screen.fill((0, 0, 0))
    #Draw the apple
    pygame.draw.rect(screen, (255, 0, 0), 
                     pygame.Rect(env.apple_position[0], env.apple_position[1], cell_size, cell_size))
    #Draw the snake
    for i, pos in enumerate(env.snake_position):
        color = (0, 255, 255) if i == 0 else (0, 255, 0)
        pygame.draw.rect(screen, color, pygame.Rect(pos[0], pos[1], cell_size, cell_size))
    pygame.display.flip()





#------------Game loop------------#
while not done:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_LEFT:
                action = 0
            elif event.key == pygame.K_RIGHT:
                action = 1
            elif event.key == pygame.K_DOWN:
                action = 2
            elif event.key == pygame.K_UP:
                action = 3
            elif event.key == pygame.K_ESCAPE:
                done = True

    #Environment step
    obs, reward, terminated, truncated, info = env.step(action)
    done = done or terminated or truncated

    #Drawing
    draw_env(env, screen)

    #Save observation + action
    human_data.append((obs.copy(), action))

    clock.tick(10) #Change FPS





#------------Save human data------------#
os.makedirs(os.path.join(os.path.dirname(__file__), "..", "data"), exist_ok=True)
with open(os.path.join(os.path.dirname(__file__), "..", "data", "human_data.pkl"), "wb") as f:
    pickle.dump(human_data, f)

print(f"Saved {len(human_data)} human transactions")
pygame.quit()
