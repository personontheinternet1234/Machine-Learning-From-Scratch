from tkinter import *
import cv2
from PIL import Image, ImageDraw


def main():
    app = Tk()
    app.geometry("280x280")

    def get_coords(event):
        global lastx, lasty
        lastx, lasty = event.x, event.y

    def draw(event):
        global lastx, lasty
        canvas.create_oval((event.x, event.y, event.x + 15, event.y + 15), fill='white', width=0)
        draw_on_image(event.x, event.y)
        lastx, lasty = event.x, event.y

    def draw_on_image(x, y):
        draw = ImageDraw.Draw(image)
        draw.ellipse((x, y, x + 23, y + 23), fill='white', width=0)

    def save_image():
        image.save("etc/drawing.png")
        print("saved")
        app.destroy()

    # create a button to save the drawing
    save_button = Button(app, text="Save", command=save_image)
    save_button.pack()

    canvas = Canvas(app, bg='black')
    canvas.pack(anchor='nw', fill='both', expand=0)

    canvas.bind("<Button-1>", get_coords)
    canvas.bind("<B1-Motion>", draw)

    # create a blank image for drawing
    image = Image.new("RGB", (280, 280), "black")

    # create frame for user drawing
    app.mainloop()

    # load the original image, and then blur it to increase likeness to MNIST dataset
    test_drawing = cv2.blur(cv2.imread('etc/drawing.png'), (10, 10))

    # resize the image to 28x28 pixels using cubic interpolation
    resized_image = cv2.resize(test_drawing, (28, 28), interpolation=cv2.INTER_CUBIC)
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)

    cv2.imwrite('etc/resized.png', grayscale_image)
    return grayscale_image


if __name__ == '__main__':
    main()
