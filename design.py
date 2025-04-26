from tkinter import Tk, Canvas, Entry, Text, Button, PhotoImage,StringVar,Label,Frame
import os



def resizer(x,mode):
    if mode=='x':
        return (x*window_size['x'])/1000
    else:
        return (x*window_size['y'])/1000
def paths(x):
    return os.path.join(os.getcwd(),x)

print(paths('button.png'))
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
students_button_image=PhotoImage(master=frame_start,file=paths('button.png'))
students_button = Button(
    frame_start,
    image=students_button_image,
    borderwidth=0,
    highlightthickness=0,
    command=lambda:print('students'),
    relief="flat",
    bg='white'
)
students_button.place(
    x=resizer(362,'x'),
    y=resizer(115,'y'),
    width=resizer(245,'x'),
    height=resizer(245,'y')
)
window.resizable(False, False)
window.mainloop()