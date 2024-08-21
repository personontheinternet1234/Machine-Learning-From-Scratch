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
BLUE = (0, 105, 148)  # Water color

# Set up the display
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
pygame.display.set_caption("Coconut Chronicles")

# Load high-quality textures (placeholder for actual textures)
player_texture = pygame.Surface((50, 50))
player_texture.fill(GREEN)

enemy_texture = pygame.Surface((50, 50))
enemy_texture.fill(RED)

coconut_texture = pygame.Surface((30, 30))
coconut_texture.fill(BROWN)

golden_coconut_texture = pygame.Surface((30, 30))
golden_coconut_texture.fill(GOLD)

# Player attributes
player_radius = 25
player_speed = 5
player_jump_speed = -10
player_velocity_y = 0
is_jumping = False

# Enemy attributes
enemy_radius = 30
enemy_speed = 3

# Coconut attributes
coconut_radius = 15

# Font for text
font = pygame.font.Font(None, 36)

def reset_game():
    global player_pos, coconuts, golden_coconut, enemy_pos, score, text_mode, text_lines, lore_index, question_mode, player_velocity_y, is_jumping
    player_pos = [SCREEN_WIDTH // 2, SCREEN_HEIGHT - player_radius]
    coconuts = [pygame.Rect(random.randint(0, SCREEN_WIDTH - 2 * coconut_radius), random.randint(0, SCREEN_HEIGHT - 2 * coconut_radius), 2 * coconut_radius, 2 * coconut_radius) for _ in range(5)]
    golden_coconut = None
    enemy_pos = [random.randint(0, SCREEN_WIDTH - 2 * enemy_radius), random.randint(0, SCREEN_HEIGHT - 2 * enemy_radius)]
    score = 0
    text_mode = True
    question_mode = False
    lore_index = 0
    player_velocity_y = 0
    is_jumping = False
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

def apply_gravity():
    global player_velocity_y, is_jumping
    if player_pos[1] < SCREEN_HEIGHT - player_radius:
        player_velocity_y += GRAVITY
    else:
        player_velocity_y = 0
        is_jumping = False

def jump():
    global player_velocity_y, is_jumping
    if not is_jumping:
        player_velocity_y = player_jump_speed
        is_jumping = True

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
                    jump()

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

        # Apply gravity
        apply_gravity()
        player_pos[1] += player_velocity_y

        # Boundary check for player
        player_pos[0] = max(player_radius, min(player_pos[0], SCREEN_WIDTH - player_radius))
        player_pos[1] = min(player_pos[1], SCREEN_HEIGHT - player_radius)

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

        # Clear the screen with water color
        screen.fill(BLUE)

        # Draw player
        screen.blit(player_texture, (player_pos[0] - player_radius, player_pos[1] - player_radius))

        # Draw enemy
        screen.blit(enemy_texture, (enemy_pos[0] - enemy_radius, enemy_pos[1] - enemy_radius))

        # Draw coconuts
        for coconut in coconuts:
            screen.blit(coconut_texture, (coconut.x, coconut.y))

        # Draw golden coconut
        if golden_coconut:
            screen.blit(golden_coconut_texture, (golden_coconut.x, golden_coconut.y))

        # Draw score
        score_text = font.render(f"Score: {score}", True, BLACK)
        screen.blit(score_text, (10, 10))

    # Update the display
    pygame.display.flip()

# Quit Pygame
pygame.quit()