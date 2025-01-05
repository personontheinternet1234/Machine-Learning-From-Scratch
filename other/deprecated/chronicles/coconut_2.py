import pygame
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
SCREEN_WIDTH = 800
SCREEN_HEIGHT = 600

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

# Create the screen object
screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

# Game clock
clock = pygame.time.Clock()

# Define the player character (Koko)
class Koko(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(WHITE)
        self.rect = self.image.get_rect()
        self.rect.center = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
        self.speed = 5
        self.elemental_powers = {
            "earth": False,
            "wind": False,
            "fire": False,
            "water": False
        }
        self.health = 100

    def update(self, keys):
        if keys[pygame.K_LEFT]:
            self.rect.x -= self.speed
        if keys[pygame.K_RIGHT]:
            self.rect.x += self.speed
        if keys[pygame.K_UP]:
            self.rect.y -= self.speed
        if keys[pygame.K_DOWN]:
            self.rect.y += self.speed

        # Placeholder for activating elemental powers
        if keys[pygame.K_e]:
            self.use_elemental_power("earth")
        if keys[pygame.K_w]:
            self.use_elemental_power("wind")
        if keys[pygame.K_f]:
            self.use_elemental_power("fire")
        if keys[pygame.K_a]:
            self.use_elemental_power("water")

    def use_elemental_power(self, element):
        if self.elemental_powers[element]:
            print(f"{element.capitalize()} power activated!")
        else:
            print(f"You haven't unlocked {element} power yet!")

# Define basic NPCs (Islanders)
class NPC(pygame.sprite.Sprite):
    def __init__(self, x, y, color):
        super().__init__()
        self.image = pygame.Surface((50, 50))
        self.image.fill(color)
        self.rect = self.image.get_rect()
        self.rect.topleft = (x, y)

    def interact(self):
        # Placeholder for dialogue or quests
        print("Greetings, Koko! There is much to be done to save the island.")

# Define a basic level layout
class Level:
    def __init__(self):
        self.background = pygame.Surface(screen.get_size())
        self.background.fill(BLUE)
        self.platforms = pygame.sprite.Group()
        self.npcs = pygame.sprite.Group()
        self.create_level()

    def create_level(self):
        # Creating some basic platforms
        for i in range(5):
            platform = pygame.Surface((200, 20))
            platform.fill(GREEN)
            platform_rect = platform.get_rect()
            platform_rect.topleft = (i * 160, SCREEN_HEIGHT - 100 - i * 40)
            self.platforms.add(platform_rect)

        # Adding some NPCs
        npc1 = NPC(300, SCREEN_HEIGHT - 150, RED)
        npc2 = NPC(600, SCREEN_HEIGHT - 200, RED)
        self.npcs.add(npc1, npc2)

    def draw(self, surface):
        surface.blit(self.background, (0, 0))
        for platform_rect in self.platforms:
            pygame.draw.rect(surface, GREEN, platform_rect)
        self.npcs.draw(surface)

# Define the Lore Codex (Placeholder)
class LoreCodex:
    def __init__(self):
        self.entries = {}

    def add_entry(self, title, content):
        self.entries[title] = content

    def display_entry(self, title):
        if title in self.entries:
            print(f"Lore Entry: {title}")
            print(self.entries[title])
        else:
            print("Lore entry not found!")

# Main game loop
def game_loop():
    # Create the character
    koko = Koko()
    all_sprites = pygame.sprite.Group()
    all_sprites.add(koko)

    # Create the level
    level = Level()

    # Create the Lore Codex
    lore_codex = LoreCodex()
    lore_codex.add_entry("The Great Shattering", "This is the tale of the island's fall from grace...")

    running = True
    while running:
        # Handle events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_l:
                    lore_codex.display_entry("The Great Shattering")

        # Get key states
        keys = pygame.key.get_pressed()

        # Update character
        all_sprites.update(keys)

        # Draw everything
        level.draw(screen)
        all_sprites.draw(screen)

        # Check for NPC interaction (placeholder logic)
        for npc in level.npcs:
            if koko.rect.colliderect(npc.rect):
                npc.interact()

        # Flip the screen buffer
        pygame.display.flip()

        # Cap the frame rate
        clock.tick(60)

    pygame.quit()

# Run the game
game_loop()
