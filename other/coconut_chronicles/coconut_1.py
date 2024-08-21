import pygame
import random

# Initialize Pygame
pygame.init()

# Constants
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)
BROWN = (139, 69, 19)
GOLD = (255, 215, 0)
BLUE = (135, 206, 235)
DARK_GRAY = (50, 50, 50)

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Coconut Chronicles")

# Player attributes
player_radius = 25
player_speed = 5

# Enemy attributes
enemy_radius = 30
enemy_speed = 3

# Coconut attributes
coconut_radius = 15

# Font for text
font = pygame.font.Font(None, 36)

# Biomes
biomes = ["Coconut Grove", "Coconut Cave", "Coconut Beach"]
current_biome = "Coconut Grove"

def reset_game():
    global player_pos, coconuts, golden_coconut, enemy_pos, score, text_mode, text_lines, lore_index, question_mode, current_biome
    player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2]
    coconuts = [pygame.Rect(random.randint(0, SCREEN_WIDTH - 2 * coconut_radius), random.randint(0, SCREEN_HEIGHT - 2 * coconut_radius), 2 * coconut_radius, 2 * coconut_radius) for _ in range(5)]
    golden_coconut = None
    enemy_pos = [random.randint(0, SCREEN_WIDTH - 2 * enemy_radius), random.randint(0, SCREEN_HEIGHT - 2 * enemy_radius)]
    score = 0
    text_mode = True
    question_mode = False
    lore_index = 0
    current_biome = "Coconut Grove"
    text_lines = [
        "Welcome to Coconut Chronicles!",
        "In the heart of the Pacific Ocean lies Coconut Island.",
        "Once home to the Arawani, a civilization blessed by the Great Coconut Spirit.",
        "A curse turned the island's guardians into vengeful spirits.",
        "Your quest: collect magical coconuts and lift the curse.",
        "Beware of the island's guardian!",
        "Press SPACE to switch to exploration mode."
    ]

# Initialize game state
reset_game()

# Lore snippets and questions
lore_snippets = [
    "The Arawani were known for their harmony with nature.",
    "The Great Coconut Spirit granted them prosperity.",
    "A sorcerer's greed led to the island's downfall.",
    "The coconuts hold the key to breaking the curse.",
    "The guardian spirits were once protectors of the island."
]

questions = [
    ("What is the name of the civilization that lived here?", "Arawani"),
    ("What did the Great Coconut Spirit grant the Arawani?", "prosperity"),
    ("What led to the island's downfall?", "greed"),
    ("What holds the key to breaking the curse?", "coconuts"),
    ("Who were the guardians before the curse?", "spirits")
]

def change_biome():
    global current_biome
    current_biome = random.choice(biomes)

# Main game loop
running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                if text_mode:
                    # If in text mode and game over, reset the game
                    if "Game Over" in text_lines:
                        reset_game()
                    else:
                        text_mode = False
                elif question_mode:
                    # Exit question mode
                    question_mode = False
                    text_mode = False
                else:
                    text_mode = True

    if text_mode:
        # Display text-based adventure
        screen.fill(WHITE)
        for i, line in enumerate(text_lines):
            text_surface = font.render(line, True, BLACK)
            screen.blit(text_surface, (20, 20 + i * 40))
    elif question_mode:
        # Display question
        screen.fill(WHITE)
        question, answer = questions[lore_index - 1]
        question_text = font.render(question, True, BLACK)
        screen.blit(question_text, (20, 20))
        answer_text = font.render(f"Answer: {answer}", True, BLACK)
        screen.blit(answer_text, (20, 60))
        instruction_text = font.render("Press SPACE to continue", True, BLACK)
        screen.blit(instruction_text, (20, 100))
    else:
        # Exploration mode
        keys = pygame.key.get_pressed()
        if keys[pygame.K_LEFT]:
            player_pos[0] -= player_speed
        if keys[pygame.K_RIGHT]:
            player_pos[0] += player_speed
        if keys[pygame.K_UP]:
            player_pos[1] -= player_speed
        if keys[pygame.K_DOWN]:
            player_pos[1] += player_speed

        # Boundary check for player
        player_pos[0] = max(player_radius, min(player_pos[0], SCREEN_WIDTH - player_radius))
        player_pos[1] = max(player_radius, min(player_pos[1], SCREEN_HEIGHT - player_radius))

        # Move the enemy randomly
        enemy_pos[0] += random.choice([-enemy_speed, enemy_speed])
        enemy_pos[1] += random.choice([-enemy_speed, enemy_speed])

        # Boundary check for enemy
        enemy_pos[0] = max(enemy_radius, min(enemy_pos[0], SCREEN_WIDTH - enemy_radius))
        enemy_pos[1] = max(enemy_radius, min(enemy_pos[1], SCREEN_HEIGHT - enemy_radius))

        # Check for coconut collection
        player_rect = pygame.Rect(player_pos[0] - player_radius, player_pos[1] - player_radius, 2 * player_radius, 2 * player_radius)
        collected_coconuts = [coconut for coconut in coconuts if player_rect.colliderect(coconut)]
        for coconut in collected_coconuts:
            coconuts.remove(coconut)
            score += 1
            # Add a new coconut at a random location
            new_coconut = pygame.Rect(random.randint(0, SCREEN_WIDTH - 2 * coconut_radius), random.randint(0, SCREEN_HEIGHT - 2 * coconut_radius), 2 * coconut_radius, 2 * coconut_radius)
            coconuts.append(new_coconut)
            if lore_index < len(lore_snippets):
                text_lines.append(lore_snippets[lore_index])
                lore_index += 1
                question_mode = True
                text_mode = True
            # Change biome occasionally
            if random.random() < 0.1:  # 10% chance to change biome
                change_biome()

        # Check for golden coconut collection
        if golden_coconut and player_rect.colliderect(golden_coconut):
            score += 10  # Golden coconut is worth more points
            golden_coconut = None  # Remove the golden coconut after collection

        # Occasionally spawn a golden coconut
        if not golden_coconut and random.random() < 0.01:  # 1% chance per frame
            golden_coconut = pygame.Rect(random.randint(0, SCREEN_WIDTH - 2 * coconut_radius), random.randint(0, SCREEN_HEIGHT - 2 * coconut_radius), 2 * coconut_radius, 2 * coconut_radius)

        # Check for collision with enemy
        enemy_rect = pygame.Rect(enemy_pos[0] - enemy_radius, enemy_pos[1] - enemy_radius, 2 * enemy_radius, 2 * enemy_radius)
        if player_rect.colliderect(enemy_rect):
            text_lines = ["You've been caught by the island's guardian!", "Game Over!", "Press SPACE to return to the main menu."]
            text_mode = True

        # Clear the screen based on the current biome
        if current_biome == "Coconut Grove":
            screen.fill(GREEN)
        elif current_biome == "Coconut Cave":
            screen.fill(DARK_GRAY)
        elif current_biome == "Coconut Beach":
            screen.fill(BLUE)

        # Draw player
        pygame.draw.circle(screen, WHITE, player_pos, player_radius)

        # Draw enemy
        pygame.draw.circle(screen, RED, enemy_pos, enemy_radius)

        # Draw coconuts
        for coconut in coconuts:
            pygame.draw.circle(screen, BROWN, coconut.center, coconut_radius)

        # Draw golden coconut
        if golden_coconut:
            pygame.draw.circle(screen, GOLD, golden_coconut.center, coconut_radius)

        # Draw score
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

        # Draw current biome
        biome_text = font.render(f"Biome: {current_biome}", True, BLACK)
        screen.blit(biome_text, (10, 40))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()