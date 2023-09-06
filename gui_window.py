#-----------------------------------------------------------------------------
# File Name : gui_window.py
# Author: Niklas Groß
#
# Creation Date : Aug 30, 2023
#
# Copyright (C) 2023 IDS, RWTH Aachen University
# Licence : GPLv3
#-----------------------------------------------------------------------------

import numpy as np

import settings
import result_figure

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

#Import the required Libraries
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from tkinter import filedialog

from customtkinter import *

#Create an instance of Tkinter frame
from PIL import ImageTk, Image

class App(CTk):
    def __init__(self, title, size):
        super().__init__()
        self.title(title)
        self.geometry(f'{size[0]}x{size[1]}')
        self.minsize(size[0], size [1])
        
        self.canva_frame = Canva_Frame(self)
        self.menu_frame = Menu_Frame(self, self.canva_frame)

        # layout
        self.rowconfigure(0, weight = 1)
        self.columnconfigure(0, weight = 4, uniform='a')
        self.columnconfigure(1, weight = 9, uniform='a')

        self.bind_all('<Control-s>', self.save_figure)
        # self.bind_all("<Control-o>", self.load_file_path)
        self.bind_all('<Control-r>', self.canva_frame.reset_figure_size)
        self.bind('<MouseWheel>', self.mousewheel_event_function)
        self.bind('<Button>', self.mousewheel_event_function)
        self.bind_all('<Return>', self.menu_frame.update_menu_frame)

        self.connect_entries()
        self.initialize_entry_values()
        self.menu_frame.update_menu_frame()

        self.mainloop ()

    def find_parent_frame(self,widget):
        parent = widget.master
        if parent == self.menu_frame:
            return self.menu_frame
        elif parent == self.canva_frame:
            return self.canva_frame
        elif parent == "":
            return None
        else:
            return self.find_parent_frame(parent)

    ##
    # function uses the settings.connected_settings list to pair the ctk entries of the connected setting
    def connect_entries(self):
        for connected_settings in settings.connected_settings:
            for setting_elem in connected_settings:
                connected_settings_copy = connected_settings.copy()
                connected_settings_copy.remove(setting_elem)
                if isinstance(setting_elem, list):
                    entry = self.menu_frame.channel_settings_frame.segment_frame[setting_elem[0]].subsegment[setting_elem[1]].elem_entry
                    entry.bind('<FocusOut>', lambda event, arg1=entry, arg2=connected_settings_copy: self.update_connected_entries(event, arg1, arg2))
                elif isinstance(setting_elem, str):
                    entry = self.menu_frame.settings_frame.settings_entries[setting_elem]
                    entry.bind('<FocusOut>', lambda event, arg1=entry, arg2=connected_settings_copy: self.update_connected_entries(event, arg1, arg2))

    def update_connected_entries(self, event, changed_entry, connected_entries):
        for setting_elem in connected_entries:
            if isinstance(setting_elem, list):
                entry = self.menu_frame.channel_settings_frame.segment_frame[setting_elem[0]].subsegment[setting_elem[1]].elem_entry
            elif isinstance(setting_elem, str):
                entry = self.menu_frame.settings_frame.settings_entries[setting_elem]

            changed_entry_value = changed_entry.get()
            if " " in changed_entry_value:
                changed_value, changed_unit = changed_entry_value.split(" ")
                if changed_unit[0] in ['k', 'M', 'G']:
                    changed_unit_prafix = changed_unit[0]
                else:
                    changed_unit_prafix = ""
            else:
                changed_value = changed_entry_value

            setting_elem_entry_value = entry.get()
            entry.delete(0, END)
            if " " in setting_elem_entry_value:
                setting_elem_value, setting_elem_unit = setting_elem_entry_value.split(" ")
                if setting_elem_unit[0] in ['k', 'M', 'G']:
                    setting_elem_unit = changed_unit_prafix+setting_elem_unit[1:]
                elif changed_unit_prafix is not None:
                    setting_elem_unit = changed_unit_prafix+setting_elem_unit
                entry.insert(0, changed_value + " " + setting_elem_unit)
            else:
                entry.insert(0, changed_value)

    def initialize_entry_values(self):
        self.menu_frame.settings_frame.initialize_entry_values()
        self.menu_frame.channel_settings_frame.initialize_entry_values()

    ##
    # function responds to the mousewheel events to scroll and zoom
    def mousewheel_event_function(self,event):
        widget = self.winfo_containing(event.x_root, event.y_root)

        if widget == self:
            return

        if widget == self.canva_frame or widget == self.menu_frame:
            parent_widget = widget
        else:
            parent_widget = self.find_parent_frame(widget)

        if parent_widget == self.canva_frame and (event.state == 0 or event.state == 16):
            self.canva_frame.canvas.xview_scroll(event.delta, "units")
            if event.num == 4:
                self.canva_frame.canvas.xview_scroll(-3, "units")
            elif event.num == 5:
                self.canva_frame.canvas.xview_scroll(3, "units")

        elif parent_widget == self.menu_frame and (event.state == 0 or event.state == 16):
            if self.menu_frame.tabview.get() == "General":
                current_frame = self.menu_frame.settings_frame
            elif self.menu_frame.tabview.get() == "Channels":
                current_frame = self.menu_frame.channel_settings_frame

            if event.num == 4:
                current_frame._parent_canvas.yview("scroll", -3, "units")
            elif event.num == 5:
                current_frame._parent_canvas.yview("scroll", 3, "units")

        elif event.state == 4 or event.state == 20: #4 is ctrl
            if event.delta > 0 or event.num == 5:
                self.canva_frame.decrease_x_scale(event)
            else:
                self.canva_frame.increase_x_scale(event)

    def save_figure(self, event):
        file_path = filedialog.asksaveasfilename(
            defaultextension='.pdf',
            filetypes=[
                ("Text-files", "*.txt"),
                ("SVG-files", "*.svg"),
                ("PDF-files", "*.pdf"),
                ("PNG-files", "*.png"),
                ("JPG-files", "*.jpg")
            ]
        )

        if file_path:
            self.canva_frame.result_figure.save_plot(file_path)
        else:
            print("Save canceled.")

    def load_file_path(self, event = None):
        try:
            file_path = filedialog.askopenfilename(filetypes=[("Text-files", "*.txt")])
            if not file_path:
                print("Open canceled.")
                return

            file_extension = os.path.splitext(file_path)[1]

            if file_extension != ".txt":
                raise Exception()

            self.open_file(os.path.relpath(file_path))

        except Exception as e:
            messagebox.showerror("ValueError", f"Unable to find or open <{file_path}>")

    def open_file(self, file_path):
        settings.initialize_variables(file_path, settings.no_gui, settings.print_info)
        self.initialize_entry_values()
        for segment_frame_index, frames in self.menu_frame.channel_settings_frame.segment_frame.items():
            current_task = settings.info_dataframe.at[segment_frame_index,'Task']
            if settings.info_dataframe.at[segment_frame_index, 'visualize']:
                self.canva_frame.result_figure.ax_data_rate[current_task].set_visible(True)
                self.canva_frame.result_figure.ax_ylabel_boxes[current_task].set_visible(True)
                self.canva_frame.result_figure.ax_ylabel_labels[current_task].set_visible(True)
            else:
                self.canva_frame.result_figure.ax_data_rate[current_task].set_visible(False)
                self.canva_frame.result_figure.ax_ylabel_boxes[current_task].set_visible(False)
                self.canva_frame.result_figure.ax_ylabel_labels[current_task].set_visible(False)

        self.menu_frame.update_menu_frame()

class Menu_Frame(CTkFrame):
    def __init__(self, parent, canva_frame):
        super().__init__(parent)
        self.parent = parent
        self.canva_frame = canva_frame

        self.grid(row = 0, column = 0, sticky = 'nsew')

        if os.path.exists(settings.logo_image_filename):
            self.image = Image.open(settings.logo_image_filename)
            self.image_frame = IMG_Frame(self, self.resize_image)
            self.image_ratio = self.image.size[0] / self.image.size[1]
            self.image_tk = ImageTk.PhotoImage(self.image)

        self.acceleration_factor_frame = self.create_subsegment("Acceleration Factor")
        self.total_latency_frame = self.create_subsegment("Total Latency")

        self.settings_label = CTkLabel(self, text = "Settings", font=settings.headline_font)
        self.settings_label.pack(side='top')

        self.tabview = CTkTabview(self)
        self.tabview.pack(side='top', fill=BOTH, expand=True)
        self.tabview.add("General")
        self.tabview.add("Channels")
        self.tabview.add("Keybinds")
        self.settings_frame = Settings_Frame(self.tabview.tab("General"))
        self.channel_settings_frame = Channel_settings_frame(self.tabview.tab("Channels"), self)
        self.ctrl_s_frame = self.create_keybind_subsegment("Ctrl-S", "Save")
        # self.ctrl_o_frame = self.create_keybind_subsegment("Ctrl-O", "Open")
        self.ctrl_mousewheel_frame = self.create_keybind_subsegment("Ctrl-Mousewheel", "Zoom")
        self.ctrl_r_frame = self.create_keybind_subsegment("Ctrl-R", "Reset Zoom")
        self.ctrl_s_frame.pack(side='top', fill=X)
        # self.ctrl_o_frame.pack(side='top', fill=X)
        self.ctrl_mousewheel_frame.pack(side='top', fill=X)
        self.ctrl_r_frame.pack(side='top', fill=X)

        self.button_frame = CTkFrame(master = self)
        self.button_frame.rowconfigure(0, weight = 1, uniform='a')
        self.button_frame.columnconfigure((0,1), weight = 1, uniform='a')
        self.generate_button = CTkButton(self.button_frame, text="Generate", command=self.update_menu_frame)
        self.generate_button.grid(row = 0, column = 0, sticky = 'nsew', padx = 4, pady = 4)
        self.reset_button = CTkButton(self.button_frame, text="Reset", command=self.reset)
        self.reset_button.grid(row = 0, column = 1, sticky = 'nsew', padx = 4, pady = 4)
        self.button_frame.pack(side='top', fill=X)

        self.total_latency_frame.pack(side='bottom', fill=X)
        self.acceleration_factor_frame.pack(side='bottom', fill=X)

        self.task_Information_Frame = Task_Information_Frame(self)
        self.latency_Information_Frame = Latency_Information_Frame(self)

    def resize_image(self,event):
        
        canvas_ratio = event.width / event.height
        
        if canvas_ratio > self.image_ratio: # canvas is wider than the image
            image_height = int(event.height)
            image_width = int(image_height * self.image_ratio)
        else: # canvas is taller than the image
            image_width = int(event.width)
            image_height = int(event.width / self.image_ratio)

        self.image_frame.delete('all')
        resized_image = self.image.resize((image_width, image_height))
        self.image_tk = ImageTk.PhotoImage(resized_image)
        self.image_frame.create_image(event.width / 2, event.height / 2, image = self.image_tk)

        self.image_frame.configure(height=image_height)

    def create_subsegment(self, elem_text):
        frame = CTkFrame(master = self)

        frame.rowconfigure(0, weight = 1)
        frame.columnconfigure((0,1), weight = 1)

        frame.elem_label = CTkLabel(frame, text = elem_text, font=settings.sub_headline_font)
        frame.elem_label.grid(row = 0, column = 0, sticky = W, pady = 2)

        frame.elem_value = CTkLabel(frame, font=settings.sub_headline_font)
        frame.elem_value.grid(row = 0, column = 1, pady = 2, sticky = 'nsew')

        return frame

    def create_keybind_subsegment(self, keybind, action):
        frame = CTkFrame(master = self.tabview.tab("Keybinds"))

        frame.rowconfigure(0, weight = 1, uniform='a')
        frame.columnconfigure((0,1), weight = 1, uniform='a')

        frame.keybind = CTkLabel(frame, text = keybind)
        frame.keybind.grid(row = 0, column = 0, sticky = E, pady = 2, padx = 4)

        frame.action = CTkLabel(frame, text = action)
        frame.action.grid(row = 0, column = 1, pady = 2, sticky = W, padx = 4)

        return frame

    def update_info_frames(self, acceleration_factor, total_latency):
        self.acceleration_factor_frame.elem_value.configure(text=round(acceleration_factor, 3))
        self.total_latency_frame.elem_value.configure(text=str(round(total_latency, 3)) + " ns")

    def reset(self, event = None):
        self.parent.open_file(settings.current_file_path)

    def update_menu_frame(self,event = None):
        self.focus_set()
        if hasattr(self.canva_frame.result_figure, 'simulation_elem'):
            self.canva_frame.reset_status()
            self.update()
        self.settings_frame.update_settings_frame_variables()
        self.channel_settings_frame.update_channel_settings_frame_variables()

        settings.update_variables()
        self.canva_frame.create_result_figure()
        self.update_info_frames(self.canva_frame.get_acceleration_factor(), self.canva_frame.result_figure.simulation_elem.total_latency)

class IMG_Frame(Canvas):
    def __init__(self, parent, resize_image):
        super().__init__(parent)
        self.pack(side='top', fill=X, expand=False)
        self.bind('<Configure>', resize_image)

class Settings_Frame(CTkScrollableFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.pack(side='top', fill=BOTH, expand=True)
        
        self.settings_entries = {}

        for elem_text, elem_setting in settings.bio_constants.items():
            frame = self.create_segment(elem_text, elem_setting)
            self.settings_entries[elem_text] = frame.elem_entry
            frame.pack(side='top')
        
        for elem_text, elem_setting in settings.network_size_dict.items():
            frame = self.create_segment(elem_text, elem_setting)
            self.settings_entries[elem_text] = frame.elem_entry
            frame.pack(side='top')

        hardware_label = CTkLabel(self, text = "Hardware", font=settings.sub_headline_font)
        hardware_label.pack(side='top')
        for settings_index, value in settings.hardware_settings.items():
            frame = self.create_segment(settings_index, value)
            self.settings_entries[settings_index] = frame.elem_entry
            frame.pack(side='top')

        latency_label = CTkLabel(self, text = "Latencies", font=settings.sub_headline_font)
        latency_label.pack(side='top')
        for settings_index, value in settings.latency.items():
            frame = self.create_segment(settings_index, value)
            self.settings_entries[settings_index] = frame.elem_entry
            frame.pack(side='top')

    def create_segment(self, elem_text, elem_setting):
        frame = CTkFrame(master = self)
        
        frame.rowconfigure(0, weight = 1, uniform='a')
        frame.columnconfigure((0,1), weight = 1, uniform='a')
        
        frame.elem_label = CTkLabel(frame, text = elem_text)
        frame.elem_label.grid(row = 0, column = 0, sticky = W, pady = 2)
        
        frame.elem_entry = CTkEntry(frame)
        frame.elem_entry.grid(row = 0, column = 1, pady = 2, sticky = 'nsew')
        
        return frame

    def initialize_entry_values(self):
        for latency_index, value in settings.latency.items():
            self.settings_entries[latency_index].delete(0, END)
            self.settings_entries[latency_index].insert(0,value)
        for settings_index, value in settings.hardware_settings.items():
            self.settings_entries[settings_index].delete(0, END)
            self.settings_entries[settings_index].insert(0,value)
        for elem_text, elem_setting in settings.bio_constants.items():
            self.settings_entries[elem_text].delete(0, END)
            self.settings_entries[elem_text].insert(0,elem_setting)
        for elem_text, elem_setting in settings.network_size_dict.items():
            self.settings_entries[elem_text].delete(0, END)
            self.settings_entries[elem_text].insert(0,elem_setting)

    def update_settings_frame_variables(self):
        try:
            network_height = int(self.settings_entries['Network Height'].get())
            network_width = int(self.settings_entries['Network Width'].get())
            network_depth = int(self.settings_entries['Network Depth'].get())
            network_4Dim = int(self.settings_entries['Network 4Dim'].get())

        except ValueError as e:
            messagebox.showerror("ValueError", "Network size settings\n"+str(e))

        try:
            self.check_network_size_dimension(network_height, network_width, network_depth, network_4Dim)

            for elem_text, elem_setting in settings.bio_constants.items():
                value = self.settings_entries[elem_text].get()
                settings.bio_constants[elem_text] = type(elem_setting)(value)

            for elem_text, elem_setting in settings.network_size_dict.items():
                settings.network_size_dict[elem_text] = int(self.settings_entries[elem_text].get())

            for elem_text, elem_setting in settings.hardware_settings.items():
                value = self.settings_entries[elem_text].get()
                settings.hardware_settings[elem_text] = type(elem_setting)(value)
            
            for latency_index in settings.latency.keys():
                settings.latency[latency_index] = float(self.settings_entries[latency_index].get())
        except ValueError as e:
            messagebox.showerror("ValueError", "General Settings\n"+str(e))

    ##
    # function checks the input for the Network_size
    def check_network_size_dimension(self, network_height, network_width, network_depth, network_4Dim):
        if network_height <= 0:
            raise ValueError("Network Height must be positive!")
        if network_width <= 0:
            raise ValueError("Network Width must be positive!")
        if network_depth <= 0:
            raise ValueError("Network Depth must be positive!")
        if network_4Dim <= 0:
            raise ValueError("Network 4Dim must be positive!")

        temp = [network_height > 1,
                network_width  > 1,
                network_depth  > 1,
                network_4Dim   > 1]

        firstIndex = next((index for index, value in enumerate(temp) if value == False), -1)
        if firstIndex != -1:
            temp = temp[firstIndex+1:]
            firstIndex = next((index for index, value in enumerate(temp) if value == True), -1)
            if firstIndex != -1:
                raise ValueError("Lower dimensions must be used first!")

class Channel_settings_frame(CTkScrollableFrame):
    def __init__(self, parent, menu_frame):
        super().__init__(parent)
        self.parent = parent
        self.menu_frame = menu_frame
        self.pack(side='top', fill=BOTH, expand=True)

        self.var = {}
        self.header_frame = {}
        self.settings_label = {}
        self.checkbox = {}
        
        self.list_extern = {}
        self.segment_frame = {}
        for info_df_index, row in settings.info_dataframe[settings.configurable_channel_settings + ['visualize']].iterrows():

            self.segment_frame[info_df_index]=self.create_segment(info_df_index, row)
            self.segment_frame[info_df_index].pack(side='top', fill=X)

        for segment_frame_index, frames in self.segment_frame.items():
            self.update_channel_axis_visualization(segment_frame_index, frames.var.get())
        
    def create_segment(self, index, row):
        segment_frame = CTkFrame(master = self)
        segment_frame.rowconfigure((0,1,2,3,4), weight = 1, uniform='a')
        segment_frame.columnconfigure(0, weight = 1, uniform='a')
        
        #HEADER
        segment_frame.header_frame = CTkFrame(segment_frame)
        segment_frame.settings_label = CTkLabel(segment_frame.header_frame, text = index, font=settings.sub_headline_font)
        segment_frame.settings_label.pack(side='left', fill=BOTH, expand=1)
        segment_frame.var = BooleanVar(value=row['visualize'])
        segment_frame.checkbox = CTkSwitch(segment_frame.header_frame, text="", variable=segment_frame.var, command= lambda: self.update_channel_axis_visualization(index, segment_frame.var.get())).pack(side='right')
        segment_frame.header_frame.grid(row = 0, column = 0, sticky = 'nsew')
            
        #LIST
        segment_frame.subsegment = {}
        for setting_index, configurable_channel_setting in enumerate(settings.configurable_channel_settings):
            segment_frame.subsegment[configurable_channel_setting] = self.create_subsegment(segment_frame, configurable_channel_setting)
            segment_frame.subsegment[configurable_channel_setting].grid(row = (setting_index+1), column = 0, sticky = 'nsew')

        if index == 'DRAM to Spike Dispatcher':
            segment_frame.subsegment['Data Size'].elem_entry.configure(state= "disabled")
        return segment_frame

    def create_subsegment(self, parent, elem_text):
        frame = CTkFrame(master = parent)
        
        frame.rowconfigure(0, weight = 1, uniform='a')
        frame.columnconfigure((0,1), weight = 1, uniform='a')
        
        frame.elem_label = CTkLabel(frame, text = elem_text)
        frame.elem_label.grid(row = 0, column = 0, sticky = W, pady = 2)
        
        frame.elem_entry = CTkEntry(frame)
        frame.elem_entry.grid(row = 0, column = 1, pady = 2, sticky = 'nsew')
        
        return frame

    def initialize_entry_values(self):
        for index, row in settings.info_dataframe[settings.configurable_channel_settings + ['visualize']].iterrows():
            self.segment_frame[index].var.set(row['visualize'])
            for setting_index, configurable_channel_setting in enumerate(zip(row[:-1], settings.configurable_channel_settings)):
                self.segment_frame[index].subsegment[configurable_channel_setting[1]].elem_entry.delete(0, END)
                self.segment_frame[index].subsegment[configurable_channel_setting[1]].elem_entry.insert(0,configurable_channel_setting[0])

    def update_channel_settings_frame_variables(self):
        try:
            for index, frames in self.segment_frame.items():
                for col_index, value in frames.subsegment.items():
                    if index != 'DRAM to Spike Dispatcher' or col_index != 'Data Size':
                        temp = value.elem_entry.get()
                        settings.info_dataframe.at[index, col_index] = type(settings.info_dataframe.at[index, col_index])(temp)
        except ValueError as e:
            messagebox.showerror("ValueError", "Channels Settings\n"+str(e))
            
    def update_channel_axis_visualization(self, index, checkboxValue):
        settings.info_dataframe.at[index, 'visualize'] = checkboxValue
        self.menu_frame.canva_frame.canva_update_axis_visualization(index, checkboxValue)

class Canva_Frame(ttk.Frame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.grid(row = 0, column = 1, columnspan=1, sticky = 'nesw')

        self.result_figure = result_figure.Result_figure()

        self.canvas = Canvas(self, borderwidth=0, highlightthickness=0)
        self.plot_frame = Frame(self.canvas)
        self.h_scrollbar = Scrollbar(self, orient=HORIZONTAL, command=self.canvas.xview)
        self.canvas.configure(xscrollcommand=self.h_scrollbar.set)

        self.canvas.pack(side=TOP, fill=BOTH, expand=1)

        self.canvas.create_window((0, 0), window=self.plot_frame, anchor='nw')
        self.bind("<Configure>", self.on_frame_configure)

        self.plot_canvas = FigureCanvasTkAgg(self.result_figure.fig, master=self.plot_frame)
        self.plot_canvas_widget = self.plot_canvas.get_tk_widget()
        self.plot_canvas_widget.pack(fill=BOTH, expand=1)

        self.status = []
        self.width = 0

    ##
    # function adjusts the canvas to the size of the window
    def on_frame_configure(self, event):
        self.height = self.winfo_height()
        self.result_figure.fig.set_size_inches(self.winfo_width()/200,self.height/200)
        self.canvas.create_window((0,0),
                                  window = self.plot_frame,
                                  anchor = 'nw',
                                  width = self.winfo_width(),
                                  height = self.height)
        self.width = self.winfo_width()
        self.canvas.configure(scrollregion = (0, 0, self.winfo_width(), self.height))

        if hasattr(self.result_figure, 'simulation_elem'):
            self.result_figure.name_Bars()

    def increase_x_scale(self, event):
        prev_width = self.width
        self.width += 0.2 * self.winfo_width()
        self.width = int(self.width)
        self.result_figure.padding_left *= prev_width/self.width
        self.result_figure.padding_right = 1-(1-self.result_figure.padding_right) * prev_width/self.width
        self.result_figure.bbox_padding_x *= prev_width/self.width
        self.result_figure.bbox_width = self.result_figure.padding_left - 3 * self.result_figure.bbox_padding_x
        self.result_figure.fig.set_size_inches(self.width/200,self.height/200)
        self.canvas.create_window((0,0),
                                  window = self.plot_frame,
                                  anchor = 'nw',
                                  width = self.width,
                                  height = self.height)
        if self.width > self.winfo_width():
            self.h_scrollbar.pack(side=BOTTOM, fill=X)
            self.result_figure.padding_bottom = settings.padding_bottom + self.h_scrollbar.winfo_height()/self.winfo_height()
            self.canvas.bind_all('<Left>', lambda event: self.canvas.xview_scroll(-1, "units"))
            self.canvas.bind_all('<Right>', lambda event: self.canvas.xview_scroll(1, "units"))
        self.canvas.configure(scrollregion = (0, 0, self.width, self.height))

        self.result_figure.update_axes()
        self.plot_canvas.draw()

    def decrease_x_scale(self, event):
        if self.width > self.winfo_width():
            prev_width = self.width
            self.width = max(self.winfo_width(), self.width - 0.2 * self.winfo_width())
            self.width = int(self.width)
            self.result_figure.padding_left *= prev_width/self.width
            self.result_figure.padding_right = 1-(1-self.result_figure.padding_right) * prev_width/self.width
            self.result_figure.bbox_padding_x *= prev_width/self.width
            self.result_figure.bbox_width = self.result_figure.padding_left - 3 * self.result_figure.bbox_padding_x
            self.result_figure.fig.set_size_inches(self.width/200,self.height/200)
            self.canvas.create_window((0,0),
                                    window = self.plot_frame,
                                    anchor = 'nw',
                                    width = self.width,
                                    height = self.height)
            if self.width <= self.winfo_width():
                self.h_scrollbar.pack_forget()
                self.result_figure.padding_bottom = settings.padding_bottom
                self.canvas.unbind_all('<Left>')
                self.canvas.unbind_all('<Right>')
            self.canvas.configure(scrollregion = (0, 0, self.width, self.height))

            self.result_figure.update_axes()
            self.plot_canvas.draw()

    ##
    # function removes all labels of the bars in the graph
    def reset_name_bars(self):
        for index, name_label in self.result_figure.ax_text_tasks.items():
            if name_label is not None:
                name_label.set_visible(False)
        self.result_figure.ax_text_tasks = {}

        for index, name_label in self.result_figure.ax_text_latencies.items():
            if name_label is not None:
                name_label.set_visible(False)
        self.result_figure.ax_text_latencies = {}

    def reset_figure_size(self, event):
        if self.width > self.winfo_width():
            self.result_figure.padding_left = settings.padding_left
            self.result_figure.padding_right = settings.padding_right
            self.result_figure.padding_top = settings.padding_top
            self.result_figure.padding_bottom = settings.padding_bottom
            self.result_figure.bbox_padding_x = settings.bbox_padding
            self.result_figure.bbox_padding_y = settings.bbox_padding
            self.result_figure.bbox_width = settings.bbox_width
            self.result_figure.name_bar_padding = settings.name_bar_padding

            self.width = self.winfo_width()
            self.result_figure.fig.set_size_inches(self.width/200,self.height/200)
            self.canvas.create_window((0,0),
                                    window = self.plot_frame,
                                    anchor = 'nw',
                                    width = int(self.width),
                                    height = self.height)
            self.h_scrollbar.pack_forget()

            self.canvas.unbind_all('<Left>')
            self.canvas.unbind_all('<Right>')
            self.canvas.configure(scrollregion = (0, 0, self.width, self.height))
            self.result_figure.update_axes()
            self.plot_canvas.draw()

    def create_result_figure(self):
        self.result_figure.generate_results()

        self.plot_canvas.draw_idle()

        for rect in self.result_figure.rects:
            rect.set_picker(True)
        for rect in self.result_figure.latency_rects:
            rect.set_picker(True)

        self.status = list((False,)*len(self.result_figure.rects + self.result_figure.latency_rects))
        self.result_figure.fig.canvas.mpl_connect('pick_event', self.onmouseover)

    def canva_update_axis_visualization(self, index, bool_value):
        current_task = settings.info_dataframe.at[index,'Task']
        if bool_value:
            self.result_figure.ax_data_rate[current_task].set_visible(True)
            self.result_figure.ax_ylabel_boxes[current_task].set_visible(True)
            self.result_figure.ax_ylabel_labels[current_task].set_visible(True)
        else:
            self.result_figure.ax_data_rate[current_task].set_visible(False)
            self.result_figure.ax_ylabel_boxes[current_task].set_visible(False)
            self.result_figure.ax_ylabel_labels[current_task].set_visible(False)

        self.result_figure.update_axes()
        self.plot_canvas.draw_idle()

        if hasattr(self.result_figure, 'simulation_elem'):
            if bool_value:
                for index, dataframe_entry in self.result_figure.simulation_elem.timing_dataframe.loc[self.result_figure.simulation_elem.timing_dataframe['Task'] == current_task][['Task', 'Start', 'End', 'color']].iterrows():
                    self.result_figure.ax_figure_lines[index] = ( self.result_figure.ax_figure.axvline(dataframe_entry['Start'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3), self.result_figure.ax_figure.axvline(dataframe_entry['End'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3) )
                    self.result_figure.ax_data_rate_lines[index] = (self.result_figure.ax_data_rate[dataframe_entry['Task']].axvline(dataframe_entry['Start'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3),
                                                    self.result_figure.ax_data_rate[dataframe_entry['Task']].axvline(dataframe_entry['End'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3) )
            else:
                for index, dataframe_entry in self.result_figure.simulation_elem.timing_dataframe.loc[self.result_figure.simulation_elem.timing_dataframe['Task'] == current_task][['Task', 'Start', 'End', 'color']].iterrows():
                    for i in range(2):
                        self.result_figure.ax_figure_lines[index][i].remove()
                        self.result_figure.ax_data_rate_lines[index][i].remove()
    ##
    # function reacts on click events on the bars in the graph to highlight or unhighlight them
    def onmouseover(self, event):
        #returns if the function is triggered by the mousewheel
        if event.guiEvent.num == 4 or event.guiEvent.num == 5:
            return

        rect = event.artist

        if rect in self.result_figure.rects:
            index = self.result_figure.rects.index(rect)
            rects_to_blur_list = [rect for rect in self.result_figure.rects if rect != self.result_figure.rects[index]] + [rect for dict_index, rect in self.result_figure.extension_rect.items() if dict_index != index] + list(self.result_figure.latency_rects)
            current_rect = self.result_figure.rects[index]
            status_index = int(index)
            current_frame = self.parent.menu_frame.task_Information_Frame
        elif rect in self.result_figure.latency_rects:
            index = self.result_figure.latency_rects.index(rect)
            rects_to_blur_list = list(self.result_figure.rects) + [rect for dict_index, rect in self.result_figure.extension_rect.items()] + [rect for rect in self.result_figure.latency_rects if rect != self.result_figure.latency_rects[index]]
            current_rect = self.result_figure.latency_rects[index]
            status_index = len(self.result_figure.rects)+int(index)
            current_frame = self.parent.menu_frame.latency_Information_Frame

        if all(val == False for val in self.status):
            self.hightlight_rect(current_frame,index, status_index, rects_to_blur_list, current_rect)

        elif self.status[status_index]:
            self.reset_status()

        else:
            self.reset_status()
            self.hightlight_rect(current_frame,index, status_index, rects_to_blur_list, current_rect)

        self.previous_frame = current_frame
        self.parent.update()
        self.result_figure.fig.canvas.draw()
    ##
    # function fades the non-highlighted bars and outlines the hightlighted bar
    def hightlight_rect(self, current_frame,index, status_index, rects_to_blur_list, current_rect):
        current_frame.pack(side='bottom', fill=BOTH, pady=3, padx=3)

        self.status[status_index] = True
        for rect_to_blur in rects_to_blur_list:
            color = rect_to_blur.get_facecolor()
            new_alpha = color[3] * 0.5
            new_color = (*color[:3], new_alpha)
            rect_to_blur.set_color(new_color)

        current_rect.set_edgecolor(settings.hightlight_edge_color)
        current_rect.set_linewidth(0.5)

        self.highlight_top = ( self.result_figure.ax_figure.axvline(current_rect.get_x(), color=settings.hightlight_edge_color, linestyle='dashed', linewidth=0.3), self.result_figure.ax_figure.axvline(current_rect.get_x()+current_rect.get_width(), color=settings.hightlight_edge_color, linestyle='dashed', linewidth=0.3) )

        if status_index < len(self.result_figure.rects):
            self.hightlight_task(index, current_rect)
        else:
            self.hightlight_latency(index, current_rect)

    ##
    # function makes the task_information_frame appear with related information
    def hightlight_task(self, index, current_rect):
        entry = self.result_figure.simulation_elem.timing_dataframe.iloc[index]
        indexes_of_task = self.result_figure.simulation_elem.timing_dataframe.loc[self.result_figure.simulation_elem.timing_dataframe.Task == entry['Task']].index.values.tolist()
        min_index = indexes_of_task.index(index)

        if entry['Task'] == 'Transmitting Spikes':
            num_of_Nodes = settings.n_transfer[min_index]
        elif entry['Task'] == 'Neuron calculation':
            num_of_Nodes = 1
        else:
            num_of_Nodes = settings.n_vector[min_index + (self.result_figure.simulation_elem.dimension+1-len(indexes_of_task))]

        self.parent.menu_frame.task_Information_Frame.set_information_frame(entry['Task'], entry['Start'], entry['End'], entry['Duration'], num_of_Nodes)

        if entry['Task'] != 'Neuron calculation':
            self.highlight_data_rate = ( self.result_figure.ax_data_rate[entry['Task']].axvline(current_rect.get_x(), color=settings.hightlight_edge_color, linestyle='dashed', linewidth=0.3), self.result_figure.ax_data_rate[entry['Task']].axvline(current_rect.get_x()+current_rect.get_width(), color=settings.hightlight_edge_color, linestyle='dashed', linewidth=0.3) )

    ##
    # function makes the latency_Information_Frame appear with related information
    def hightlight_latency(self, index, current_rect):
        entry = self.result_figure.simulation_elem.latency_dataframe.iloc[index]
        self.parent.menu_frame.latency_Information_Frame.set_information_frame(entry['Hardwareclass'], entry['Start'], entry['End'], entry['Duration'])

    ##
    # function
    def reset_status(self):
        if True not in self.status:
            return

        first_true_index = self.status.index(True)
        for rect_reset_blur in list(self.result_figure.rects) + list(self.result_figure.latency_rects) + [rect for dict_index, rect in self.result_figure.extension_rect.items()]:
            color = rect_reset_blur.get_facecolor()
            new_alpha = 1
            new_color = (*color[:3], new_alpha)
            rect_reset_blur.set_color(new_color)

        self.previous_frame.pack_forget()

        if first_true_index < len(self.result_figure.rects):
            prev_task = self.result_figure.simulation_elem.timing_dataframe.at[first_true_index, 'Task']
            self.result_figure.rects[first_true_index].set_linewidth(0)
            if prev_task != 'Neuron calculation':
                self.highlight_data_rate[0].remove()
                self.highlight_data_rate[1].remove()
        else:
            self.result_figure.latency_rects[first_true_index-len(self.result_figure.rects)].set_linewidth(0)

        for i in range(2):
            self.highlight_top[i].remove()
        
        self.status[int(first_true_index)] = False

    def get_acceleration_factor(self):
        return self.result_figure.simulation_elem.acceleration_factor

class Task_Information_Frame(CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent

        self.corner_radius=0
        self.name_var = StringVar(value="")
        self.spikes_var = StringVar(value="")

        self.rowconfigure((0,1,2,3), weight = 1)
        self.columnconfigure((0,1,2), weight = 1, uniform='a')

        self.task_name_label = CTkLabel(self, textvariable = self.name_var, font=settings.sub_headline_font)
        self.task_name_label.grid(row = 0, column = 0, columnspan=4, sticky = W)

        self.spikes_label = CTkLabel(self, textvariable = self.spikes_var, justify="left", anchor="w", font=settings.standard_font)
        self.spikes_label.grid(row = 1, column = 0, columnspan=3, sticky = W)

        self.start_label = CTkLabel(self, text = "Start:", font=settings.standard_font)
        self.start_label.grid(row = 2, column = 0, sticky = W)

        self.start_label_2= CTkLabel(self, font=settings.standard_font)
        self.start_label_2.grid(row = 3, column = 0, sticky = W)

        self.end_label = CTkLabel(self, text = "End:", font=settings.standard_font)
        self.end_label.grid(row = 2, column = 1, sticky = W)

        self.end_label_2= CTkLabel(self, font=settings.standard_font)
        self.end_label_2.grid(row = 3, column = 1, sticky = W)

        self.duration_label = CTkLabel(self, text = "Duration:", font=settings.standard_font)
        self.duration_label.grid(row = 2, column = 2, sticky = W)

        self.duration_label_2= CTkLabel(self, font=settings.standard_font)
        self.duration_label_2.grid(row = 3, column = 2, sticky = W)

    ##
    # function updates the information to be displayed in the Task_Information_Frame
    def set_information_frame(self, task, start, end, duration, num_of_nodes):
        self.name_var.set(task)

        self.start_label_2.configure(text = str(round(start, 2)) + " ns")
        self.end_label_2.configure(text = str(round(end, 2)) + " ns")
        self.duration_label_2.configure(text = str(round(duration, 2)) + " ns")

        num_of_spikes = np.array(num_of_nodes) * settings.spikesPerNodePerTimestep

        if isinstance(num_of_spikes, np.float64):
            self.spikes_var.set(str(round(num_of_spikes, 2)) + " Spikes processed in this task.")
        elif isinstance(num_of_spikes, np.ndarray):
            direction = ["x", "y", "z", "4D"]
            text = "\n".join(["{: <6} {: <40}".format(*(round(value, 2), f"spikes transmitted in {dir}-direction")) for value, dir in zip(list(num_of_spikes), direction)])
            self.spikes_var.set(text)

class Latency_Information_Frame(CTkFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.name_var = StringVar(value="")

        self.rowconfigure((0,1,2,3), weight = 1)
        self.columnconfigure((0,1,2), weight = 1, uniform='a')

        self.task_name_label = CTkLabel(self, textvariable = self.name_var, font=settings.sub_headline_font)
        self.task_name_label.grid(row = 0, column = 0, columnspan=4, sticky = W)

        self.start_label = CTkLabel(self, text = "Start:", font=settings.standard_font)
        self.start_label.grid(row = 1, column = 0, sticky = W)

        self.start_label_2= CTkLabel(self, font=settings.standard_font)
        self.start_label_2.grid(row = 2, column = 0, sticky = W)

        self.end_label = CTkLabel(self, text = "End:", font=settings.standard_font)
        self.end_label.grid(row = 1, column = 1, sticky = W)

        self.end_label_2= CTkLabel(self, font=settings.standard_font)
        self.end_label_2.grid(row = 2, column = 1, sticky = W)

        self.duration_label = CTkLabel(self, text = "Duration:", font=settings.standard_font)
        self.duration_label.grid(row = 1, column = 2, sticky = W)

        self.duration_label_2= CTkLabel(self, font=settings.standard_font)
        self.duration_label_2.grid(row = 2, column = 2, sticky = W)

    ##
    # function updates the information to be displayed in the Latency_Information_Frame
    def set_information_frame(self, task, start, end, duration):
        self.name_var.set(task + " – Latency")

        self.start_label_2.configure(text = str(round(start, 2)) + " ns")
        self.end_label_2.configure(text = str(round(end, 2)) + " ns")
        self.duration_label_2.configure(text = str(round(duration, 2)) + " ns")