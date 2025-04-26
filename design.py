from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage,StringVar,Label,Frame
import os
from functions import paths , resizer 
import cv2
from threading import Thread
from PIL import Image, ImageTk
from functions import student_menue



    



window = Tk()
window.attributes("-fullscreen", True)

window_size = {'x': window.winfo_screenwidth() , 'y': window.winfo_screenheight()}
window.title('Mositto')
frame_start=Frame(window)
frame_start.pack()
canvas = Canvas(
    frame_start,
    bg = "#FFFFFF",
    height = resizer(1000,'y'),
    width = resizer(1000,'x'),
    bd = 0,
    highlightthickness = 0,
    relief = "ridge"
)
canvas.place(x = 0 , y = 0)
canvas.pack()
canvas.create_rectangle(0, 0, resizer(1000,'x'), resizer(85,'y'), fill="#38b6ff", outline="")
logo=PhotoImage(master=frame_start , file=paths('image.png'))
label = Label(frame_start, image=logo , bg='#38b6ff')
label.image = logo
label.place(x=resizer(200,'x'),y=0)
students_button_image=PhotoImage(master=frame_start,file=paths('button.png'))
students_button = Button(
    frame_start,
    image=students_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=student_menue,
    relief="flat",
    bg='white'
)
students_button.place(
    x=resizer(362,'x'),
    y=resizer(115,'y'),
    width=resizer(245,'x'),
    height=resizer(245,'y')
)

teachers_button_image=PhotoImage(master=frame_start,file=paths('button_1.png'))
teachers_button = Button(
    frame_start,
    image=teachers_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:print('teachers'),
    relief="flat",
    bg='white'
)
teachers_button.place(
    x=resizer(362,'x'),
    y=resizer(421,'y'),
    width=resizer(245,'x'),
    height=resizer(245,'y')
)

managers_button_image=PhotoImage(master=frame_start,file=paths('button_2.png'))
managers_button = Button(
    frame_start,
    image=managers_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:print('managers'),
    relief="flat",
    bg='white'
)
managers_button.place(
    x=resizer(362,'x'),
    y=resizer(727,'y'),
    width=resizer(245,'x'),
    height=resizer(245,'y')
)

window.resizable(False, False)
window.mainloop()