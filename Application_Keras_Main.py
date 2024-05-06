"""
MNIST Neural Network Application
Isaac Park Verbrugge & Christian SW Host-Madsen
"""

import ast
import numpy as np
import pygame
import time

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

# window settings
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
font = pygame.font.Font(None, 36)

# set window title
pygame.display.set_caption("Drawing Window")
# fill background
screen.fill(background_colour)
# update display
pygame.display.flip()

# run pygame
running = True
# frame rate checker
clock = pygame.time.Clock()

# run pygame window
while running:
    # frame rate
    clock.tick(frame_rate)
    for event in pygame.event.get():
        # show directions
        # rendered_text = font.render("C to clear", True, (0, 255, 0))
        # screen.blit(rendered_text, (0, 0))
        # rendered_text = font.render("S to evaluate", True, (0, 255, 0))
        # screen.blit(rendered_text, (0, 40))
        # pygame.display.flip()
        # check if mouse down
        if pygame.mouse.get_pressed()[0]:
            # find mouse
            m_x, m_y = event.pos
            # draw circle on mouse
            pygame.draw.circle(screen, (255, 255, 255), (m_x, m_y), 10)
            # update display
            pygame.display.flip()
        # check if key pressed
        if event.type == pygame.KEYDOWN:
            # clear screen
            if event.key == pygame.K_c:
                screen.fill(background_colour)
                pygame.display.flip()
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

                # show output on pygame window
                rendered_text = font.render(text, True, (0, 255, 0))
                screen.blit(rendered_text, (0, 0))
                pygame.display.flip()

                # print technical output in python terminal
                for i in range(len(list(output[0]))):
                    print(f"{i}: {output[0][i]:.5f}")
                print("-------------------------")
                # wait until acknowledged
                # while not pygame.mouse.get_pressed()[0]:
                #     time.sleep(1 / frame_rate)
                # screen.fill(background_colour)
                # pygame.display.flip()
        # closed window
        if event.type == pygame.QUIT:
            # end running program
            running = False
