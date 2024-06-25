"""
MNIST Number Drawing
Authors:
    Christian SW Host-Madsen CO '25 <chost-madsen25@punahou.edu>
    Isaac Park Verbrugge CO '25 <iverbrugge25@punahou.edu>
"""

import ast
import numpy as np
import pygame
import os
from Garden.Extra import Credits
from Garden.Models.NeuralNetwork import NeuralNetwork
import Garden.Functions.Functional as Fnc
import Garden.Functions.Formatter as Fmr
import Garden.Functions.Metrics as Mtr

from PIL import Image

""" definitions """


# softmax activator
def softmax(values):
    output = np.exp(values) / np.sum(np.exp(values))
    return output


def guess(network):
    # save image
    scaled_screen = pygame.transform.scale(screen, (win_length / win_scale, win_height / win_scale))
    pygame.image.save(scaled_screen, f"assets/other/{image_location}")
    # open saved image
    img = Image.open(f"assets/other/{image_location}")
    # grayscale image
    gray_img = img.convert("L")
    # convert to numpy array
    forward_layer = np.array(list(gray_img.getdata())) / 255

    # forward pass and argmax of image
    output = network.forward(forward_layer)[-1][0]
    text = str(np.nanargmax(output))
    # softmax output
    smax_output = softmax(output)

    # show output on pygame window
    rendered_text = font.render(text, True, (0, 255, 0))
    screen.blit(rendered_text, (win_length - 10, 5))
    pygame.display.flip()
    # wait until input to clear screen
    wait_clear = True
    rendered_error = False

    # print certainties in terminal
    for i in range(len(list(smax_output))):
        print(f"{i}: {(smax_output[i] * 100):.5f}%")
    print("-------------------------")


""" load files """

# file locations
weights_location = "weights_keras.txt"
biases_location = "biases_keras.txt"
image_location = "user_number.jpeg"

# load weights and biases
weights = []
biases = []
with open(f"assets/saved/{weights_location}", "r") as f:
    for line in f:
        weights.append(np.array(ast.literal_eval(line)))
with open(f"assets/saved/{biases_location}", "r") as f:
    for line in f:
        biases.append(np.array(ast.literal_eval(line)))

layer_sizes = [784, 16, 16, 10]
network = NeuralNetwork(weights=weights, biases=biases, layer_sizes=layer_sizes, activation="leaky relu")

""" application """

# initialize pygame
pygame.init()

# background color
background_colour = (0, 0, 0)
# window scale
win_scale = 40
win_height = 28 * win_scale
win_length = 28 * win_scale
screen = pygame.display.set_mode((win_length, win_height))
# frame rate
frame_rate = 60
# font type
font = pygame.font.Font(None, 30)

# set window title
pygame.display.set_caption("Drawing Window")
# fill background
screen.fill(background_colour)
# update display
pygame.display.flip()

# pygame running
running = True
# frame rate checker
clock = pygame.time.Clock()
# mouse position list
mouse_pos = []
# display preset conditionals
wait_initial = True
rendered_initial = False
wait_clear = False
rendered_error = False

# print initial text break
print("-------------------------")

if not os.path.exists(os.path.dirname(os.path.join('assets', 'other', 'user_number.jpeg'))):
    os.mkdir(os.path.dirname(os.path.join('assets', 'other', 'user_number.jpeg')))

WHITE = (255, 255, 255)
GREY = (200, 200, 200)
BLACK = (0, 0, 0)


def draw_text(text, font, color, x, y):
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.center = (x, y)
    screen.blit(text_surface, text_rect)


def draw_button(text, font, color, rect):
    pygame.draw.rect(screen, WHITE, rect)
    pygame.draw.rect(screen, GREY, rect, width=3)
    draw_text(text, font, color, rect.centerx, rect.centery)


button_check = pygame.Rect(0, win_height - (win_length/10) - 5, win_length/5, win_length/10)

# run pygame window
while running:
    # frame rate
    clock.tick(frame_rate)

    for event in pygame.event.get():
        # show directions
        if wait_initial:
            # if not already rendered
            if not rendered_initial:
                # render all texts
                screen.fill(background_colour)
                rendered_text = font.render("C to clear", True, (0, 255, 0))
                screen.blit(rendered_text, (0, 0))
                rendered_text = font.render("S to save & evaluate", True, (0, 255, 0))
                screen.blit(rendered_text, (0, 20))
                rendered_text = font.render("D to reshow directions", True, (0, 255, 0))
                screen.blit(rendered_text, (0, 40))
                # update screen
                pygame.display.flip()
                # signal rendered
                rendered_initial = True
            # wait until mouse down
            if pygame.mouse.get_pressed()[0]:
                # clear and end waiting loop
                wait_initial = False
                screen.fill(background_colour)
                pygame.display.flip()
        # waiting for next user input after evaluation
        if wait_clear:
            # wait until mouse down
            if pygame.mouse.get_pressed()[0]:
                # clear and end waiting loop
                wait_clear = False
                screen.fill(background_colour)
                pygame.display.flip()
        # check if mouse down
        if pygame.mouse.get_pressed()[0]:
            x, y = pygame.mouse.get_pos()
            if button_check.collidepoint(x, y):
                guess(network)
                break

            # find mouse and reference last pos
            mouse_pos.append(event.pos)
            mouse_pos.append(event.pos)
            mouse_pos[1] = mouse_pos[0]
            mouse_pos[0] = event.pos
            # draw circle on mouse
            pygame.draw.circle(screen, (255, 255, 255), mouse_pos[0], 40)
            # draw connecting line
            pygame.draw.line(screen, (255, 255, 255), mouse_pos[1], mouse_pos[0], 84)
            mouse_pos = mouse_pos[0:2]
            # update display
            pygame.display.flip()
        # check if mouse not down
        if not pygame.mouse.get_pressed()[0]:
            # reset mouse positions
            mouse_pos = []
        # check if key pressed
        if event.type == pygame.KEYDOWN:
            # clear screen
            if event.key == pygame.K_c:
                screen.fill(background_colour)
                pygame.display.flip()
            # set conditionals for program directions
            if event.key == pygame.K_d:
                wait_initial = True
                rendered_initial = False
            # evaluate image
            if event.key == pygame.K_s:
                if not wait_initial and not wait_clear:
                    guess(network)
                else:
                    if not rendered_error:
                        rendered_text = font.render("No new image to evaluate", True, (0, 255, 0))
                        screen.blit(rendered_text, (0, 260))
                        pygame.display.flip()
                        rendered_error = True

        draw_button("Guess?", font, BLACK, button_check)
        pygame.display.flip()

        # closed window
        if event.type == pygame.QUIT:
            running = False
