import tkinter as tk
from tkinter import ttk
from tkinter import EventType
import math

def draw(event):
    x, y = event.x, event.y
    if canvas.old_coords:
        x1, y1 = canvas.old_coords
        canvas.create_line(x, y, x1, y1)
    canvas.old_coords = x, y

def create_circle(x, y, r, canvas, fill=None): #center coordinates, radius
    x0 = x - r
    y0 = y - r
    x1 = x + r
    y1 = y + r
    return canvas.create_oval(x0, y0, x1, y1, fill=fill)

class Spawn:
    def __init__(self, root) -> None:
        self.slider = tk.Scale(
            root,
            from_=0,
            to=100,
            orient='horizontal',  # horizontal
        )

        self.label = ttk.Label(root, text = "Probabily Spawn") 

        # all the spawn areas belong to one group
        # it will also have one object or group of objects assigned
        groups  = [f'Group {i}' for i in range(5)]
        self.group = tk.StringVar()
        self.current_group = ttk.Combobox(root, width = 27, textvariable = self.group)
        
        # Adding combobox drop down list
        self.current_group['values'] = groups

        self.current_group.bind('<<ComboboxSelected>>', self.group_changed)

        self.colours = {group:colour for group, colour in zip(groups, ["white", "black", "red", "green", "blue", "cyan", "yellow"])}
        self.current_colour = self.colours['Group 1']

    def show(self):
        self.slider.pack()
        self.label.pack()
        self.current_group.pack(padx=5, pady=5)

    def hide(self):
        self.slider.pack_forget()
        self.label.pack_forget()
        self.current_group.pack_forget()

    def group_changed(self, event):
        """ handle the mode changed event """
        choosen_group = self.group.get()
        print(f'Choosen group {choosen_group}')
        self.current_colour = self.colours[choosen_group]
        
    def draw_rectangle(self, event):
        if event.type == EventType.ButtonPress:
            x = event.x
            y = event.y
            canvas.init_rectangle = (x, y)
        elif event.type == EventType.Motion:
            if canvas.temp_square != None:
                canvas.delete(canvas.temp_square)
            x, y = canvas.init_rectangle
            canvas.temp_square = canvas.create_rectangle(x, y, event.x, event.y, fill=self.current_colour)
            
        elif event.type == EventType.ButtonRelease:
            x, y = canvas.init_rectangle
            canvas.create_rectangle(x, y, event.x, event.y, fill=self.current_colour)

# https://stackoverflow.com/questions/47996285/how-to-draw-a-line-following-your-mouse-coordinates-with-tkinter
def draw_line(event):
    # print(event.__dict__.items())
    # print(event.type)
    
    radius = 10
    
    if str(event.type) == EventType.ButtonPress:
        print("---------------")
        x = event.x
        y = event.y
        
        canvas.old_coords = event.x, event.y
        print(f'checking {(x, y)}')
        
        create_circle(x, y, radius, canvas)
        for point in canvas.existing_points:
            center_x = point[0]
            center_y = point[1]
            print((x-center_x)**2 + (y - center_y)**2 < radius**2)
            if ((x-center_x)**2 + (y - center_y)**2) < radius**2:
                canvas.old_coords = center_x, center_y
                print(f'In in {(center_x, center_y)} {(x, y)} | {((x-center_x)^2 + (y - center_y)^2) < radius^2}')      

    elif str(event.type) == EventType.Motion:
        canvas.itemconfigure(mouse_pos, text=f'x:{event.x} y:{event.y} | x: {event.x-200} y: {(-event.y)+200}')
        if canvas.prev != None:
            canvas.delete(canvas.prev)

        x1, y1 = canvas.old_coords

        # this line needs to be straight, 0, 45, 90 degrees
        vector = (event.x - x1, event.y - y1)
        # print(f'Vector {vector}')
        try:
            angle = math.degrees(math.atan(vector[1]/vector[0]))
        except ZeroDivisionError:
            angle = 90.0
        # print(f'({event.x} {event.y}) ({x1} {y1}) vector {vector} degrees {angle}')

        print(angle)

        # y
        if angle < 20 and angle > -20:
            canvas.prev = canvas.create_line(x1, y1, event.x, y1)
            canvas.final_point = (event.x, y1)
        elif abs(angle) > 80:
            canvas.prev = canvas.create_line(x1, y1, x1, event.y)
            canvas.final_point = (x1, event.y)


    elif str(event.type) == EventType.ButtonRelease:
        x, y = canvas.final_point
        x1, y1 = canvas.old_coords

        create_circle(x, y, radius, canvas)

        canvas.existing_points.add((x, y))
        canvas.existing_points.add(canvas.final_point)
        canvas.init = None
        canvas.create_line(x, y, x1, y1)

        # create and aux list to delete
        canvas.lines.append(([x-200, (-y)+200], [x1-200, (-y1)+200]))

def reset_coords(event):
    canvas.old_coords = None     



root = tk.Tk()

canvas_width = 400
canvas_height = 400
canvas = tk.Canvas(root, bg='white', width=canvas_width, height=canvas_height)

canvas.pack(side = tk.RIGHT)
canvas.prev = None
canvas.init = None
canvas.old_coords = None
canvas.existing_points = set()
canvas.lines = list()

canvas.init_rectangle = None
canvas.temp_square = None
canvas.rectangles = list()

mouse_pos = canvas.create_text(40,40, text="-", fill="darkblue")

#  exit button
deploy_btn = ttk.Button(
    root,
    text='Deploy',
    command=lambda: root.destroy()
)

deploy_btn.pack(
    ipadx=5,
    ipady=5,
    expand=True
)

# dropdown option
mode = tk.StringVar()
mode_choosen = ttk.Combobox(root, width = 27, textvariable = mode)
  
# Adding combobox drop down list
mode_choosen['values'] = ('Walls', 
                          'Spawn Areas',
                          'Groups Spawn Areas')

mode_choosen.pack(padx=5, pady=5)

spawn_ui = Spawn(root=root)

def mode_changed(event):
    """ handle the mode changed event """
    print(f'Selected mode: {mode.get()}!!!!!!!')
    current_mode = mode.get()

    if current_mode == 'Walls':
        canvas.bind('<Button-1>', draw_line)
        canvas.bind('<ButtonRelease-1>', draw_line)
        canvas.bind('<B1-Motion>', draw_line)
        spawn_ui.hide()
        
    elif current_mode == 'Spawn Areas':
        canvas.bind('<Button-1>', spawn_ui.draw_rectangle)
        canvas.bind('<ButtonRelease-1>', spawn_ui.draw_rectangle)
        canvas.bind('<B1-Motion>', spawn_ui.draw_rectangle)
        # show dropdown for objects types
        # show percentage of spawn
        spawn_ui.show()

mode_choosen.bind('<<ComboboxSelected>>', mode_changed)

slider = ttk.Scale(
    root,
    from_=0,
    to=100,
    orient='horizontal',  # horizontal
)


l3 = ttk.Label(root, text = "Horizontal Scaler")

slider.pack()
l3.pack()

langs = ['Java', 'C#', 'C', 'C++', 'Python',
         'Go', 'JavaScript', 'PHP', 'Swift']

var = tk.Variable(value=langs)

listbox = tk.Listbox(
    root,
    listvariable=var,
    height=6,
    selectmode=tk.EXTENDED
)

################

# canvas.bind('<Motion>', draw_line)

center_x = canvas_width/2
center_y = canvas_height/2

create_circle(center_x, center_y, 4, canvas, "#ff0")


canvas.create_line(center_x, center_y, center_x + 40, center_y, fill="#f00")
canvas.create_line(center_x, center_y, center_x, center_y + -40, fill="#0f0")

# root.bind('<ButtonRelease-1>', reset_coords)

# root.mainloop()