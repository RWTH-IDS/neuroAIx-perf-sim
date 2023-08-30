Python Network Simulator

tested on:
MacOS Ventura 13.4.1
Ubuntu 20.04.4
Arch Linux
Windows 10 (might be slow)

Used packages:

pip install numpy pandas matplotlib Pillow customtkinter tk

numpy 1.24.3
matplotlib 3.7.1
Pillow 9.5.0

On some systems, the following installations may need to be performed with sudo:
sudo apt-get install python3-tk
sudo apt-get install python3-pil python3-pil.imagetk

No optimal representation possible with Conda
____________________________________________________________________________________________________________________________

To generate the results from Fig. 3, execute `python main.py`

positional arguments:
  file_path             Direction of settings txt file. If no file is specified, the microcircuit.txt file is loaded.

options:
  -h, --help            show this help message and exit
  --no_gui              Deactivates the GUI. By default, the GUI is turned on.
  --save_fig [SAVE_FIG ...]
                        Saves the figure if no_gui is set (optional name) (optional file extension jpg, png, svg, pdf)
  --print_info {none,standard,detailed}
                        Specifies the level of information to print. When the GUI is on, this option defaults to "none",
                        when the GUI is off, the option defaults to "standard". 
____________________________________________________________________________________________________________________________

Configuration text file:
When specifying clock, data rate and data size, it is necessary to write the unit after the numerical value. 
It is to be noted that within a stage the data rate and data size must be converted by division without further conversions of the units into a time duration.

The stages indicate the individual stations that are repeated in each time step.
The order of the stages indicates the order in which they are displayed in the graphic.
The following settings must be filled in for a stage:
index               =   Indicates where the data transfer takes place
Task                =   Specifies the name of the task
Number of Channel   =   Specifies the number of the channel
Max. Data Rate      =   Specifies the data rate at which data is transferred within this task
Data Size           =   Specifies the data size of a packet
depend              =   Specifies the name of the task from which this task appends
Added Latency       =   Specifies the name of the latencies by which this task is delayed (can also be empty).
Hardwareclass       =   Describes to which hardware class this task can be added (important for coloring)

It is possible to link settings together so that when one of the two numerical values is changed, both settings are changed.
The setting that is specified as the reference value must be defined in the hardware_settings.

Color settings do not have to be defined. Undefined hardware classes are assigned one color and undefined latencies are assigned another.
However, since in this case only two different colors are used, it is recommended to define the colors in the file.
____________________________________________________________________________________________________________________________

Usage in GUI:
Keybinds:
Ctrl-S          – Saving of the graph and the configuration
Ctrl-Mousewheel – Zooming in the graphs
Ctrl-R          – Reset zooming in the graph
