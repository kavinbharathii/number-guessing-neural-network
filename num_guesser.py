import pygame
# import math
import numpy as np
import tensorflow as tf
from tensorflow import keras

model = keras.models.load_model('num_guesser.model')

rez = 20
width = 560
height = 560
display = pygame.display.set_mode((width, height))
pygame.display.set_caption('Number Guesser')


num_arr = np.zeros(shape=(28, 28), dtype=np.float)


def get_rect_pos():
    x = pygame.mouse.get_pos()[0]
    y = pygame.mouse.get_pos()[1]
    return (x - (x % rez), y - (y % rez), rez, rez)


def get_grid_pos():
    x = pygame.mouse.get_pos()[0]
    y = pygame.mouse.get_pos()[1]
    return x // rez, y // rez


def draw_grid():
    for i in range(1, int(width / rez)):
        pygame.draw.line(display, (255, 255, 255),
                         (i * rez, 0), (i * rez, width))

    for j in range(1, int(height / rez)):
        pygame.draw.line(display, (255, 255, 255),
                         (0, j * rez), (height, j * rez))


def main():
    run = True
    while run:
        draw_grid()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False

            if pygame.mouse.get_pressed()[0]:
                pygame.draw.rect(display, (255, 255, 255), get_rect_pos())
                num_arr[get_grid_pos()[1]][get_grid_pos()[0]] = 1

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    num_arr.shape = (1, 784)
                    prediction = model.predict(num_arr)
                    num_guessed = list(prediction[0]).index(max(prediction[0]))
                    confidence = round(max(prediction[0]) * 100, 2)
                    print('\n')

                    for index, result in enumerate(prediction[0]):
                        print(
                            f'label {index}: % confidence = {round(result * 100, 2)}%\n')

                    print(
                        f'number guessed: {num_guessed}; confidence = {confidence}%')

        pygame.display.flip()


if __name__ == '__main__':
    main()
