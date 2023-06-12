import tkinter as tk
from tkinter import ttk
from tkinter import EventType
import math
import uuid

class Spawn:
    def __init__(self, root, canvas) -> None:
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
        self.current_group_combobox = ttk.Combobox(root, width = 27, textvariable = self.group)
        
        # Adding combobox drop down list
        self.current_group_combobox['values'] = groups

        self.current_group_combobox.bind('<<ComboboxSelected>>', self.group_changed)

        self.colours = {group:colour for group, colour in zip(groups, ["white", "black", "red", "green", "blue", "cyan", "yellow"])}
        self.current_colour = self.colours['Group 0']
        self.group_to_folder = {group:f"g{idx}" for group, idx in zip(groups, range(5))}

        self.root = root
        self.canvas = canvas

        self.spawn_areas = list()

    def show(self):
        self.slider.pack()
        self.label.pack()
        self.current_group_combobox.pack(padx=5, pady=5)

    def hide(self):
        self.slider.pack_forget()
        self.label.pack_forget()
        self.current_group_combobox.pack_forget()

    def group_changed(self, event):
        """ handle the mode changed event """
        choosen_group = self.group.get()
        print(f'Choosen group {choosen_group}')
        self.current_colour = self.colours[choosen_group]
        self.current_group = choosen_group
        
        
    def draw_spawn_area(self, event):
        if event.type == EventType.ButtonPress:
            x = event.x
            y = event.y
            self.canvas.init_rectangle = (x, y)
        elif event.type == EventType.Motion:
            if self.canvas.temp_square != None:
                self.canvas.delete(self.canvas.temp_square)
            x, y = self.canvas.init_rectangle
            self.canvas.temp_square = self.canvas.create_rectangle(x, y, event.x, event.y, outline=self.current_colour)
            
        elif event.type == EventType.ButtonRelease:
            x1, y1 = self.canvas.init_rectangle
            x2, y2 = event.x, event.y
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=self.current_colour)
            if x1 > x2:
                x1, x2 = x2, x1
            y1, y2 = -y1, -y2
            if y1 > y2:
                y1, y2 = y2, y1
            self.spawn_areas.append((uuid.uuid4(), x1-200, y1+200, x2-200, y2+200, self.current_group))


class UserInterface:
    def __init__(self) -> None:
        self.root = tk.Tk()

        canvas_width = 400
        canvas_height = 400
        self.canvas = tk.Canvas(self.root, bg='white', width=canvas_width, height=canvas_height)

        self.canvas.pack(side = tk.RIGHT)
        self.canvas.prev = None
        self.canvas.init = None
        self.canvas.old_coords = None
        self.canvas.existing_points = set()
        self.canvas.lines = list()

        self.canvas.init_rectangle = None
        self.canvas.temp_square = None
        self.canvas.rectangles = list()

        # for use in draw_line
        self.mouse_pos = self.canvas.create_text(40,40, text="-", fill="darkblue")

        #  exit button
        deploy_btn = ttk.Button(
            self.root,
            text='Deploy',
            command=lambda: self.root.destroy()
        )

        deploy_btn.pack(
            ipadx=5,
            ipady=5,
            expand=True
        )

        # dropdown option
        self.mode = tk.StringVar()
        mode_choosen = ttk.Combobox(self.root, width = 27, textvariable = self.mode)
        
        # Adding combobox drop down list
        mode_choosen['values'] = ('Walls', 
                                'Spawn Areas',
                                'Groups Spawn Areas')

        mode_choosen.pack(padx=5, pady=5)

        self.spawn_ui = Spawn(root=self.root, canvas=self.canvas)

        mode_choosen.bind('<<ComboboxSelected>>', self.mode_changed)

        slider = ttk.Scale(
            self.root,
            from_=0,
            to=100,
            orient='horizontal',  # horizontal
        )


        l3 = ttk.Label(self.root, text="Horizontal Scaler")

        slider.pack()
        l3.pack()

        langs = ['Java', 'C#', 'C', 'C++', 'Python',
                'Go', 'JavaScript', 'PHP', 'Swift']

        var = tk.Variable(value=langs)

        listbox = tk.Listbox(
            self.root,
            listvariable=var,
            height=6,
            selectmode=tk.EXTENDED
        )

        ################

        # self.canvas.bind('<Motion>', draw_line)

        center_x = canvas_width/2
        center_y = canvas_height/2

        self.create_circle(center_x, center_y, 4, self.canvas, "#ff0")


        self.canvas.create_line(center_x, center_y, center_x + 40, center_y, fill="#f00")
        self.canvas.create_line(center_x, center_y, center_x, center_y + -40, fill="#0f0")

        # root.bind('<ButtonRelease-1>', reset_coords)

        # root.mainloop()

    def update_UI(self):
        self.root.update_idletasks()
        self.root.update()

    def reset_coords(self, event):
        self.canvas.old_coords = None

    def get_canvas_lines(self):
        return self.canvas.lines
    
    def get_spawn_areas(self):
        return self.spawn_ui.spawn_areas

    def mode_changed(self, event):
        """ handle the mode changed event """
        print(f'Selected mode: {self.mode.get()}!!!!!!!')
        current_mode = self.mode.get()

        if current_mode == 'Walls':
            self.canvas.bind('<Button-1>', self.draw_line)
            self.canvas.bind('<ButtonRelease-1>', self.draw_line)
            self.canvas.bind('<B1-Motion>', self.draw_line)
            self.spawn_ui.hide()
            
        elif current_mode == 'Spawn Areas':
            self.canvas.bind('<Button-1>', self.spawn_ui.draw_spawn_area)
            self.canvas.bind('<ButtonRelease-1>', self.spawn_ui.draw_spawn_area)
            self.canvas.bind('<B1-Motion>', self.spawn_ui.draw_spawn_area)
            # show dropdown for objects types
            # show percentage of spawn
            self.spawn_ui.show()

    # https://stackoverflow.com/questions/47996285/how-to-draw-a-line-following-your-mouse-coordinates-with-tkinter
    def draw_line(self, event):

        radius = 10

        if event.type == EventType.ButtonPress:
            print("---------------")
            x = event.x
            y = event.y

            self.canvas.old_coords = event.x, event.y
            print(f'checking {(x, y)}')

            self.create_circle(x, y, radius, self.canvas)
            for point in self.canvas.existing_points:
                center_x = point[0]
                center_y = point[1]
                print((x-center_x)**2 + (y - center_y)**2 < radius**2)
                if ((x-center_x)**2 + (y - center_y)**2) < radius**2:
                    self.canvas.old_coords = center_x, center_y
                    print(
                        f'In in {(center_x, center_y)} {(x, y)} | {((x-center_x)^2 + (y - center_y)^2) < radius^2}')

        elif event.type == EventType.Motion:
            self.canvas.itemconfigure(
                self.mouse_pos, text=f'x:{event.x} y:{event.y} | x: {event.x-200} y: {(-event.y)+200}')
            if self.canvas.prev != None:
                self.canvas.delete(self.canvas.prev)

            x1, y1 = self.canvas.old_coords

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
                self.canvas.prev = self.canvas.create_line(x1, y1, event.x, y1)
                self.canvas.final_point = (event.x, y1)
            elif abs(angle) > 80:
                self.canvas.prev = self.canvas.create_line(x1, y1, x1, event.y)
                self.canvas.final_point = (x1, event.y)

        elif event.type == EventType.ButtonRelease:
            x, y = self.canvas.final_point
            x1, y1 = self.canvas.old_coords

            self.create_circle(x, y, radius, self.canvas)

            self.canvas.existing_points.add((x, y))
            self.canvas.existing_points.add(self.canvas.final_point)
            self.canvas.init = None
            self.canvas.create_line(x, y, x1, y1)

            # create and aux list to delete
            self.canvas.lines.append(([x-200, (-y)+200], [x1-200, (-y1)+200]))

    def draw(self, event):
        x, y = event.x, event.y
        if self.canvas.old_coords:
            x1, y1 = self.canvas.old_coords
            self.canvas.create_line(x, y, x1, y1)
        self.canvas.old_coords = x, y


    def create_circle(self, x, y, r, canvas, fill=None):  # center coordinates, radius
        x0 = x - r
        y0 = y - r
        x1 = x + r
        y1 = y + r
        return canvas.create_oval(x0, y0, x1, y1, fill=fill)
