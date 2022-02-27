from tkinter import *
import tkinter.font as tkFont
from PIL import Image, ImageDraw
import PIL
import pickle
import numpy as np
import cv2
from tensorflow.keras.models import load_model
#loaded_model = open("trained_model.p", "rb")
#model = pickle.load(loaded_model)
#
#lastx, lasty = None, None
model = load_model('model.h5')

def clear_widget():
    global draw_board, image1, draw, text
    image1 = PIL.Image.new("RGB", (600, 200), (255, 255, 255))
    text.delete(1.0, END)
    draw = ImageDraw.Draw(image1)
    draw_board.delete('all')

def draw_lines(event):
    global lastx, lasty
    x, y = event.x, event.y
    draw_board.create_line((lastx, lasty, x, y), width=8, fill='black', capstyle=ROUND, smooth=TRUE, splinesteps=12)
    draw.line([lastx, lasty, x, y], fill="black", width=10)
    lastx, lasty = x, y

def activate_event(event):
    global lastx, lasty
    draw_board.bind('<B1-Motion>', draw_lines)
    lastx, lasty = event.x, event.y




def save():
    text_num = []
    global image_number
    filename = 'image_out.png'
    image1.save(filename)
    image = cv2.imread(filename)
    gray = cv2.cvtColor(image.copy(), cv2.COLOR_BGR2GRAY)
    ret, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    contours = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(image, (x, y), (x + w, y + h), (255, 0, 0), 1)
        digit = th[y:y + h, x:x + w]
        resized_digit = cv2.resize(digit, (18, 18))
        padded_digit = np.pad(resized_digit, ((5, 5), (5, 5)), "constant", constant_values=0)
        print(padded_digit.shape)
        digit = padded_digit.reshape(1, 28, 28, 1)
        digit = digit / 255.0

        pred = model.predict([digit])[0]
        final_pred = np.argmax(pred)
        text_num.append([x, final_pred])


        data = str(final_pred) + ' ' + str(int(max(pred) * 100)) + '%'

        font = cv2.FONT_HERSHEY_SIMPLEX
        fontScale = 0.5
        color = (0, 0, 0)
        thickness = 1
        cv2.putText(image, data, (x, y - 5), font, fontScale, color, thickness)

    text_num = sorted(text_num, key=lambda t: t[0])
    text_num = [i[1] for i in text_num]
    final_text = "".join(map(str, text_num))
    text.insert(END, final_text)
    cv2.imshow('image', image)
    cv2.waitKey(0)


win = Tk()
win.geometry("650x500")
win.title("Multiple Handwritten Digit Recognition")
win.config(background="#FFC0CB")

fontStyle = tkFont.Font(family="Lucida Grande", size=15)

write_label = Label(win, text="Write your number:", bg="#DF5727", font=fontStyle)
write_label.place(relx=0.03, rely=0.03)

draw_board = Canvas(win, width=600, height=200, bg='white')
draw_board.place(relx=0.03, rely=0.1)
draw_board.bind('<Button-1>', activate_event)
#

image1 = PIL.Image.new("RGB", (600, 200), (255, 255, 255))
draw = ImageDraw.Draw(image1)

button=Button(text="Predict", command=save, bg="#66c2ff", font=tkFont.Font(family="Lucida Grande", size=20))
button.place(relx=0.5, rely=0.63, anchor=CENTER)

predict_label = Label(win, text="Predicted Number:", bg="#DF5727", font=tkFont.Font(family="Lucida Grande", size=13))
predict_label.place(relx=0.03, rely=0.7)

text = Text(win, height=2, width=25, font=tkFont.Font(family="Lucida Grande", size=13))
text.place(relx=0.03, rely=0.77)

del_btn = Button(win, text="Erase All", command=clear_widget, bg="#66c2ff", width=8, font=tkFont.Font(family="Lucida Grande", size=15))
del_btn.place(relx=0.03, rely=0.88)


win.mainloop()