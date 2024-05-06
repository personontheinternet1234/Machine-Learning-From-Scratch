"""
MNIST Neural Network Application
Isaac Park Verbrugge & Christian SW Host-Madsen
"""

import ast
import numpy as np
import pygame

from PIL import Image

""" definitions """


# leaky rectified linear activator
def l_relu(values):
    output = np.maximum(0.1 * values, values)
    return output


# forward pass
def forward(inputs, weights, biases):
    nodes = [inputs]
    # set amount of layers
    for layer in range(3):
        activations = l_relu(np.matmul(nodes[-1], weights[layer]) + biases[layer])
        nodes.append(activations)
    return nodes


# softmax activator
def softmax(values):
    return np.exp(values) / np.sum(np.exp(values))


""" load files """

# file locations
weights_location = "weights_keras.txt"
biases_location = "biases_keras.txt"
image_location = "user_number.jpeg"

# load weights and biases
weights = []
biases = []
with open(f"saved/{weights_location}", "r") as f:
    for line in f:
        weights.append(np.array(ast.literal_eval(line)))
with open(f"saved/{biases_location}", "r") as f:
    for line in f:
        biases.append(np.array(ast.literal_eval(line)))

""" application """

# initialize pygame
pygame.init()

# background color
background_colour = (0, 0, 0)
# window scale
win_height = 280
win_length = 280
downscale_factor = 0.1
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

# print initial text break
print("-------------------------")

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
            # find mouse and reference last pos
            mouse_pos.append(event.pos)
            mouse_pos.append(event.pos)
            mouse_pos[1] = mouse_pos[0]
            mouse_pos[0] = event.pos
            # draw circle on mouse
            pygame.draw.circle(screen, (255, 255, 255), mouse_pos[0], 10)
            # draw connecting line
            pygame.draw.line(screen, (255, 255, 255), mouse_pos[1], mouse_pos[0], 21)
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
                # save image
                scaled_screen = pygame.transform.scale(screen, (win_length * downscale_factor, win_height * downscale_factor))
                pygame.image.save(scaled_screen, f"saved/{image_location}")
                # open saved image
                img = Image.open(f"saved/{image_location}")
                # grayscale image
                gray_img = img.convert("L")
                # convert to numpy array
                forward_layer = np.array(list(gray_img.getdata())) / 255

                # forward pass and argmax of image
                output = forward(forward_layer, weights, biases)[-1]
                text = str(np.nanargmax(output))
                # softmax output
                smax_output = softmax(output)

                # show output on pygame window
                rendered_text = font.render(text, True, (0, 255, 0))
                screen.blit(rendered_text, (260, 260))
                pygame.display.flip()
                # wait until input to clear screen
                wait_clear = True

                # print certainties in terminal
                for i in range(len(list(smax_output[0]))):
                    print(f"{i}: {(smax_output[0][i] * 100):.5f}%")
                print("-------------------------")

        # closed window
        if event.type == pygame.QUIT:
            # end running program
            running = False
