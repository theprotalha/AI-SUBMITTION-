print("\n--- Q1: CartPole Environment using Gymnasium & Pygame ---\n")
import gymnasium as gym
import pygame

env = gym.make("CartPole-v1", render_mode="human")

font = None

for episode in range(1, 20):
    score = 0
    state, info = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

        if font is None:
            pygame.font.init()
            font = pygame.font.SysFont("Arial", 24)

        surface = pygame.display.get_surface()
        text = font.render(f"Score: {int(score)}", True, (255, 0, 0))
        surface.blit(text, (10, 10))
        pygame.display.update()

    print(f"Episode {episode} Score: {score}")

env.close()
pygame.quit()

print("____________________________________________________________________________________________________________________________________________")
print("\n--- Q2: MountainCar Environment using Gymnasium & Pygame ---\n")
import gymnasium as gym
import pygame

env = gym.make("MountainCar-v0", render_mode="human")

font = None
best_score = -float('inf')

# We only need a few episodes to prove it works with a better policy
NUM_EPISODES = 5 

for episode in range(1, NUM_EPISODES + 1):
    state, info = env.reset()
    done = False
    score = 0

    while not done:
        # Task 7/8: Advanced Rule-Based Action
        # state[1] is velocity. If velocity is moving right (>0), push right (2).
        # If moving left (<0), push left (0). This builds momentum rapidly.
        if state[1] > 0:
            action = 2
        else:
            action = 0
            
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        score += reward

        if font is None:
            pygame.font.init()
            font = pygame.font.SysFont("Arial", 24)

        surface = pygame.display.get_surface()
        text = font.render(f"Episode: {episode}  Score: {int(score)}", True, (0, 0, 255))
        surface.blit(text, (200, 20))
        
        # Reduced delay for faster execution
        pygame.time.delay(5) 
        pygame.display.update()

    print(f"Episode {episode} Score: {score}")
    if score > best_score:
        best_score = score

env.close()
pygame.quit()

print(f"\nOptimization Results:")
print(f"Best Score Achieved: {best_score}")

