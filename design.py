from tkinter import Tk, Canvas, Button, PhotoImage,Label,Frame
from functions import paths , resizer 
from functions import student_menue , atab_stu
from time import sleep

def switch_to_student_menue():
    student_menue()  # Assuming this function exists
    print(atab_stu)  # Assuming atab_stu is properly initialized and contains 'present' list

    # Create a label
    l = Label(master=frame_start)
    l.place(x=resizer(100, 'x'), y=resizer(500, 'y'))

    index = 0  # Initialize the index to track the student
    
    # Function to update the label with delay
    def update_label():
        nonlocal index
        if index < len(atab_stu['present']):
            l.config(text=f'Welcome to Mositto {atab_stu["present"][index]}')  # Update text
            index += 1
            window.after(2000, update_label)  # Schedule next update after 2 seconds
        else:
            l.config(text='')  # Clear text after all students have been displayed
            atab_stu['present'] = []  # Clear the 'present' list

    # Start the label updates
    update_label()







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
    command=switch_to_student_menue,
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