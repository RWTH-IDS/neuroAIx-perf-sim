#-----------------------------------------------------------------------------
# File Name : settings.py
# Author: Niklas Groß
#
# Creation Date : Aug 30, 2023
#
# Copyright (C) 2023 IDS, RWTH Aachen University
# Licence : GPLv3
#-----------------------------------------------------------------------------

import pandas as pd
import numpy as np
import math
import warnings
from itertools import combinations
import os
import platform

bio_constants = {}
hardware_settings = {}
network_size_dict = {}
latency = {}

#Padding settings for the graph
padding_left = 0.2
padding_right = 1 - 0.04
padding_top = 1 - 0.08
padding_bottom = 0.02
bbox_padding = 0.01
bbox_width = padding_left - 3 * bbox_padding

name_bar_padding = 4

headline_font = ("Arial", 32)
sub_headline_font = ("Arial", 19)
standard_font = ("Arial", 16)

configurable_channel_settings = ['Number of Channel', 'Max. Data Rate', 'Data Size', 'Added Latency']

supported_figure_file_extensions = ['svg', 'pdf', 'png', 'jpg', 'jpeg']
default_figure_file_extension = 'pdf'

accuracy = 1e-6

connected_settings = []

color_dict = {}
default_color_latency = '#E49B58'
default_color_stage = '#9999CC'
background_color = '#36454F'
hightlight_edge_color = '#000000'
node_box_background_color = '#F1C39D'
max_marker_color = '#7A7A7A'
duration_extension_color = '#9E0021'

logo_image_filename = "rwth_ids_bild_rgb.png"

class latency_element():
    def __init__(self, string):
        self.elements = string.split(",")
        self.elements = [elem.strip() for elem in self.elements]
        self.latency_duration = 0
        self.string = string
        if string != "":
            for elem in self.elements:
                if elem == 'Additional Hop Delay':
                    self.latency_duration += sum([i-2 for i in list(network_size_dict.values()) if i > 2]) * float(latency[elem]) / (3+3)
                else:
                    self.latency_duration += float(latency[elem])

    def __add__(self, other):
        if isinstance(other, latency_element):
            return self.latency_duration + other.latency_duration
        elif isinstance(other, (int, float)):
            return self.latency_duration + float(other)
        else:
            raise TypeError("Unsupported operand type for +.")

    def __radd__(self, other):
        if isinstance(other, latency_element):
            return self.latency_duration + other.latency_duration
        elif isinstance(other, (int, float)):
            return self.latency_duration + float(other)
        else:
            raise TypeError("Unsupported operand type for +.")

    def __sub__(self, other):
        if isinstance(other, unit_element):
            return self.latency_duration - other.latency_duration
        elif isinstance(other, (int, float)):
            return self.latency_duration - other
        else:
            raise TypeError("Unsupported operand type for -.")

    def __mul__(self, other):
        if isinstance(other, unit_element):
            return self.latency_duration * other.latency_duration
        elif isinstance(other, (int, float)):
            return self.latency_duration * other
        else:
            raise TypeError("Unsupported operand type for *.")

    def __truediv__(self, other):
        if isinstance(other, unit_element):
            return self.latency_duration / other.latency_duration
        elif isinstance(other, (int, float)):
            return self.latency_duration / other
        else:
            raise TypeError("Unsupported operand type for /.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return other / self.latency_duration
        else:
            raise TypeError("Unsupported operand type for /.")

    def __str__(self):
            return f"{self.string}"

    def __format__(self, format_spec):
        return self.__str__()

class unit_element():
    def __init__(self, string):
        merged_dict = {**bio_constants, **hardware_settings}
        value, space, unit = string.rpartition(" ")
        value = value.strip()
        self.unit = unit.strip()
        self.to_be_connected = False

        try:
            self.element_value = float(value)
        except ValueError:
            if value == 'bytes_per_synaptic_list':
                self.element_value = bytes_per_synaptic_list
            else:
                if isinstance(merged_dict[value], unit_element):
                    self.element_value = merged_dict[value].element_value
                    self.align_prefix(merged_dict[value])
                elif isinstance(merged_dict[value], (int,float)):
                    self.element_value = float(merged_dict[value])
                self.to_be_connected = True
                self.connected_name = value

        self.string = str(self.element_value) + " " + unit
        self.free_prefactors_and_set_to_ns()

    ##
    # function converts element_value to a version that has no prefix and then converts to ns if possible
    def free_prefactors_and_set_to_ns(self):
        self.element_value_prefix_free_and_ns = self.element_value
        self.unit_prefix_free_and_ns = self.unit
        if self.unit[0] == 'G':
            self.element_value_prefix_free_and_ns *= 1e9
            self.unit_prefix_free_and_ns = self.unit_prefix_free_and_ns[1:]
        elif self.unit[0] == 'M':
            self.element_value_prefix_free_and_ns *= 1e6
            self.unit_prefix_free_and_ns = self.unit_prefix_free_and_ns[1:]
        elif self.unit[0] == 'k':
            self.element_value_prefix_free_and_ns *= 1e3
            self.unit_prefix_free_and_ns = self.unit_prefix_free_and_ns[1:]

        if '/' in self.unit_prefix_free_and_ns:
            elem, time = self.unit_prefix_free_and_ns.split("/")
            if time == 's':
                self.element_value_prefix_free_and_ns *= 1e-9
            elif time == 'ms':
                self.element_value_prefix_free_and_ns *= 1e-6
            elif time == 'µs' or time == 'us':
                self.element_value_prefix_free_and_ns *= 1e-3
            self.unit_prefix_free_and_ns = elem + '/ns'
        elif self.unit_prefix_free_and_ns.endswith('Hz') or self.unit_prefix_free_and_ns.endswith('hz'):
            self.element_value_prefix_free_and_ns *= 1e-9
            self.unit_prefix_free_and_ns = self.unit_prefix_free_and_ns.replace("Hz", "/ns")

    ##
    # function adjusts the element_value to the prefix specified in the config file
    # so that the value still matches the connected unit element
    def align_prefix(self, other):
        if self.unit[0] in ['k', 'M', 'G'] and self.unit[0] == other.unit[0]:
            return
        if other.unit[0] in ['k', 'M', 'G']:
            position = ['k', 'M', 'G'].index(other.unit[0])
            power = 3 * (position+1)
            self.element_value *= 10 ** power
        if self.unit[0] in ['k', 'M', 'G']:
            position = ['k', 'M', 'G'].index(self.unit[0])
            power = 3 * (position+1)
            self.element_value /= 10 ** power

    def round(self, num_of_fractional_digits):
        return unit_element(str(round(self.element_value, num_of_fractional_digits)) + " " + self.unit)

    def __rmul__(self, other):
        if isinstance(other, (int, float)):
            return unit_element(str(self.element_value * other) + " " + self.unit)
        else:
            raise TypeError("Unsupported operand type for *.")

    def __mul__(self, other):
        if isinstance(other, (int, float)):
            return unit_element(str(self.element_value * other) + " " + self.unit)
        else:
            raise TypeError("Unsupported operand type for *.")

    def __truediv__(self, other):
        if isinstance(other, unit_element):
            self_units = [self.unit_prefix_free_and_ns]
            other_units = [other.unit_prefix_free_and_ns]
            if "/" in self.unit_prefix_free_and_ns:
                self_units = self_units[0].split("/")
            elif "/" in other.unit_prefix_free_and_ns:
                other_units = other_units[0].split("/")

            if self_units[0].lower() == other_units[0].lower() or (self_units[0] == 'B' and other_units[0] == 'Byte') or (self_units[0] == 'Byte' and other_units[0] == 'B'):
                return self.element_value_prefix_free_and_ns / other.element_value_prefix_free_and_ns
            elif (self_units[0] == 'B' or self_units[0] == 'Byte') and (other_units[0] == 'bit' or other_units[0] == 'Bit' or other_units[0] == 'b'):
                return self.element_value_prefix_free_and_ns * 8 / other.element_value_prefix_free_and_ns
            elif (self_units[0] == 'bit' or other_units[0] == 'Bit' or self_units[0] == 'b') and (other_units[0] == 'B' or other_units[0] == 'Byte'):
                return self.element_value_prefix_free_and_ns / (other.element_value_prefix_free_and_ns * 8)
            else:
                warnings.warn("There is no Unitalignment for this case", UserWarning)
                return
        elif isinstance(other, (int, float)):
            new_value = self.element_value / other
            return unit_element(str(new_value) + " " + self.unit)
        else:
            raise TypeError("Unsupported operand type for /.")

    def __rtruediv__(self, other):
        if isinstance(other, (int, float)):
            return float(other) / self.element_value_prefix_free_and_ns
        else:
            raise TypeError("Unsupported operand type for /.")

    def __str__(self):
            return f"{self.string}"

    def __format__(self, format_spec):
        return self.__str__()

##
# function links setting items and stores those that depend on each other in a list.
def add_to_connected_settings(value, index):
    global connected_settings

    if len(connected_settings) == 0:
        connected_settings = [[value, index]]
    else:
        for sublist in connected_settings:
            if value in sublist:
                sublist.append(index)
                break
            else:
                connected_settings.append([value, index])

def add_to_dataframe(index, Task, Number_of_Channel, Max_Data_Rate, Data_Size, depend, Added_Latency, Hardwareclass):
    global info_dataframe

    temp_info_dataframe = pd.DataFrame({'Task': Task,
                                        'Number of Channel': Number_of_Channel,
                                        'max. rel. usable Data_Rate': 1.0 if Task != 'Write into RingBuffer' else calculate_dram_data_rate(),
                                        'Max. Data Rate': [Max_Data_Rate],
                                        'Data Size': [Data_Size],
                                        'affect': [np.array([None])],
                                        'depend': depend,
                                        'Added Latency': Added_Latency,
                                        'visualize': True if Task != 'Requesting Lookup' else False,
                                        'Hardwareclass': Hardwareclass},
                                   index=[index])

    info_dataframe = pd.concat([info_dataframe, temp_info_dataframe])

##
# function converts the input strings into an int/float/unit_element
def convert_input_value(input_value, identifier):
    value = input_value.strip()
    if '.' in value:
        try:
            entry = float(value)
        except ValueError:
            entry = unit_element(value)
            if entry.to_be_connected:
                add_to_connected_settings(entry.connected_name, identifier)
    else:
        try:
            entry = int(value)
        except ValueError:
            pass
            entry = unit_element(value)
            if entry.to_be_connected:
                add_to_connected_settings(entry.connected_name, identifier)
    return entry

##
# function sets a default color for the latency elements and hardware class for which no color is set
def set_default_colors():
    total_stage_entries = ['Calculation'] + list(set(info_dataframe['Hardwareclass'].values.tolist()))
    total_latency_entries = ['Pipeline'] + list(latency.keys())

    color_dict.update({entry: default_color_stage for entry in total_stage_entries if entry not in color_dict})
    color_dict.update({entry: default_color_latency for entry in total_latency_entries if entry not in color_dict})

def initialize_variables(file_path, no_gui_arg, print_info_arg):
    global synapses_per_Node, bytes_per_synaptic_list, color_dict, info_dataframe, latency, task_dependency, network_size_dict, connected_settings, current_file_path, no_gui, print_info
    current_file_path = file_path
    no_gui = no_gui_arg
    print_info = print_info_arg

    info_dataframe = pd.DataFrame(columns=['Task',
                                           'Number of Channel',
                                           'max. rel. usable Data_Rate',
                                           'Max. Data Rate',
                                           'Data Size',
                                           'affect',
                                           'depend',
                                           'Added Latency',
                                           'visualize',
                                           'Hardwareclass'])

    task_dependency = {}

    if platform.system() == 'Windows':
        script_dir = os.path.dirname(__file__)
        updated_file_path = os.path.join(script_dir, file_path)
    else:
        updated_file_path = file_path

    with open(updated_file_path, 'r') as file:
        for line in file:
            line = line.strip()
            if line.startswith('bio_constants'):
                current_section = 'bio_constants'
            elif line.startswith('hardware_settings'):
                current_section = 'hardware_settings'
            elif line.startswith('network_size'):
                current_section = 'network_size'
            elif line.startswith('latency'):
                synapses_per_Node = bio_constants['Synapses per Neuron'] / np.prod(list(network_size_dict.values()))
                bytes_per_synaptic_list	= math.ceil(synapses_per_Node * hardware_settings['Bytes per synapse'] / (hardware_settings['DRAM fetch size'] / 8)) * (hardware_settings['DRAM fetch size'] / 8)
                current_section = 'latency'
            elif line.startswith('stages'):
                current_section = 'stages'
            elif line.startswith('colors'):
                current_section = 'colors'

            elif line and current_section == 'bio_constants':
                setting_elements = ['Neurons', 'Fire rate', 'Time resolution', 'Synapses per Neuron']
                identifier, value = line.split('=')
                identifier = identifier.strip()

                if identifier not in setting_elements:
                    continue
                entry = convert_input_value(value, identifier)
                bio_constants[identifier] = entry
            elif line and current_section == 'hardware_settings':
                setting_elements = ['System clock', 'Neuron clock', 'Pipeline depth', 'Number of Cores', 'DRAM fetch size', 'Bytes per synapse', 'Flitsize']
                identifier, value = line.split('=')
                identifier = identifier.strip()

                if identifier not in setting_elements:
                    continue
                entry = convert_input_value(value, identifier)
                hardware_settings[identifier] = entry
            elif line and current_section == 'network_size':
                setting_elements = ['Network Height', 'Network Width', 'Network Depth', 'Network 4Dim']
                identifier, value = line.split('=')
                identifier = identifier.strip()

                if identifier not in setting_elements:
                    continue
                entry = convert_input_value(value, identifier)
                network_size_dict[identifier] = entry
            elif line and current_section == 'latency':
                identifier, value = line.split('=')
                identifier = identifier.strip()
                entry = convert_input_value(value, identifier)
                latency[identifier] = entry
            elif line and current_section == 'stages':
                identifier, value = line.split('=')
                identifier = identifier.strip()
                value = value.strip()
                if identifier == 'index':
                    index = value
                elif identifier == 'Task':
                    Task = value
                elif identifier == 'Number of Channel':
                    Number_of_Channel = int(value)
                elif identifier == 'Max. Data Rate':
                    Max_Data_Rate = unit_element(value)
                    if Max_Data_Rate.to_be_connected:
                        add_to_connected_settings(Max_Data_Rate.connected_name, [index, 'Max. Data Rate'])
                elif identifier == 'Data Size':
                    Data_Size = unit_element(value)
                    if Data_Size.to_be_connected:
                        add_to_connected_settings(Data_Size.connected_name, [index, 'Data Size'])
                elif identifier == 'depend':
                    depend = value.strip()
                    task_dependency[index] = depend
                elif identifier == 'Added Latency':
                    Added_Latency = latency_element(value)
                elif identifier == 'Hardwareclass':
                    Hardwareclass = value
                    add_to_dataframe(index, Task, Number_of_Channel, Max_Data_Rate, Data_Size, depend, Added_Latency, Hardwareclass)
            elif line and current_section == 'colors':
                identifier, value = line.split('=')
                identifier = identifier.strip()
                value = value.strip()
                color_dict[identifier] = value

    for info_df_index, info_df_entry in info_dataframe.iterrows():
        affected_tasks = [info_dataframe.at[key, 'Task'] for key,val in task_dependency.items() if val == info_df_entry['Task']]
        if affected_tasks != []:
            info_dataframe.at[info_df_index, 'affect'] = np.array(affected_tasks)

    set_default_colors()

##
# function returns the interpolated maximum relative data rate of the DRAM
def calculate_dram_data_rate():
    dram_max_data_rate = 12.12 # (GB/s)
    # dram_data_rate = [[8, 1.296], [48, 6.1632], [96, 9.9648], [200, 12.560], [400, 15.744], [1000, 19.000], [2000, 20.400], [4000, 21.120], [8000, 21.760]]
    dram_data_rate = [[64, 0.648], [384, 3.0816], [768, 4.9824], [1600, 6.28], [3200, 7.872], [8000, 9.5], [16000, 10.200], [32000, 10.56], [64000, 10.88]]

    x_values = [point[0] for point in dram_data_rate]
    # Binary Search to find the interval
    left_index = 0
    right_index = len(x_values) - 1
    max_dram_data_rate = 0

    while left_index < right_index:
        mid_index = (left_index + right_index) // 2
            
        if x_values[mid_index] < bytes_per_synaptic_list:
            left_index = mid_index + 1
        else:
            right_index = mid_index
        
    # used linear interpolation to calculate dram_data_rate y = y1 + (x-x1) * (y2-y1)/(x2-x1)
    if bytes_per_synaptic_list > x_values[-1]:
        max_dram_data_rate = dram_data_rate[-1][1]
    else:
        max_dram_data_rate = dram_data_rate[left_index-1][1] + (bytes_per_synaptic_list - dram_data_rate[left_index-1][0]) * ((dram_data_rate[left_index][1] - dram_data_rate[left_index-1][1]) / (dram_data_rate[left_index][0] - dram_data_rate[left_index-1][0]))
    return max_dram_data_rate / dram_max_data_rate

##
# function calculates all possibilities for the distribution of the spikes distributed in the 1D line in the 2D planes
def create_vectors_1d_to_2d(vector, indexes, network_vectors):
    sub_list = [[] for i in range(2)]
    for i in range(2):
        sub_list[i] = vector.copy()
        sub_list[i][indexes[(i+1)%2]] = sub_list[i][indexes[(i+1)%2]] + network_vectors[indexes[i]]

    return sub_list

##
# function calculates all possibilities for the distribution of the spikes distributed in the 2D planes into the 3D spaces
def create_vectors_2d_to_3d(vector, indexes, network_planes):
    sub_list = [[] for i in range(3)]
    for i in range(3):
        sub_list[i] = vector.copy()
        sub_list[i][indexes[(i+2)%3]] = sub_list[i][indexes[(i+2)%3]] + network_planes[indexes[i]][indexes[(i+1)%3]]

    return sub_list

##
# function returns the vector with the smallest maximum vector element and the minimum sum of all vector elements
def get_minimal_vector(vectors):
    tuples_index_maximal_length = [(index, max(elem)) for index, elem in enumerate(vectors)]
    minimal_length = min(x[1] for x in tuples_index_maximal_length)

    indexes_with_same_minimal_length = [x[0] for x in tuples_index_maximal_length if x[1] == minimal_length]
    tuples_index_sum_of_length = [(index, sum(elem)) for index, elem in enumerate(vectors) if index in indexes_with_same_minimal_length]

    result = vectors[min(tuples_index_sum_of_length, key=lambda x: x[1])[0]]
    return result

##
# function determines for each time step the number of spikes to send to each dimension
# to minimize the data transfer per cable and returns the elements stored as a list
def calculate_transfer_list(network_size):
    network_dimensions = [x-1 for x in network_size]
    n = len(network_dimensions)

    transfer_list = [[] for i in range(n)]

    if network_dimensions[0] == 0:
        return transfer_list

    transfer_list[0] = [1 if i > 0 else 0 for i in network_dimensions]

    if network_dimensions[1] == 0:
        return transfer_list

    base_vectors = [[0 for i in range(n)]]
    for combination_elem in combinations([x for x in range(n)], 2):
        new_base_vectors = []

        for elem in base_vectors:
            new_base_vectors.extend(create_vectors_1d_to_2d(elem, combination_elem, network_dimensions))

        base_vectors = new_base_vectors.copy()

    transfer_list[1] = get_minimal_vector(base_vectors)

    if network_dimensions[2] == 0:
        return transfer_list

    network_plane_matrix = [[0 for i in range(n)] for i in range(n)]
    for index_1, index_2 in combinations([x for x in range(n)], 2):
        network_plane_matrix[index_1][index_2] = network_dimensions[index_1] * network_dimensions[index_2]
        network_plane_matrix[index_2][index_1] = network_dimensions[index_1] * network_dimensions[index_2]

    base_vectors = [[0 for i in range(n)]]
    for combination_elem in combinations([x for x in range(n)], 3):
        new_base_vectors = []

        for elem in base_vectors:
            new_base_vectors.extend(create_vectors_2d_to_3d(elem, combination_elem, network_plane_matrix))

        base_vectors = new_base_vectors.copy()

    transfer_list[2] = get_minimal_vector(base_vectors)

    if network_dimensions[3] == 0:
        return transfer_list

    network_3D_sizes = [np.prod(i) for i in combinations([x for x in network_dimensions], 3)]
    min_val = min(network_3D_sizes)
    min_index = network_3D_sizes.index(min_val)

    t = list(combinations([x for x in range(n)], 3))[min_index]
    missing_num = list(set(range(n)) - set(t))[0]

    resulting_list = [0 for i in range(n)]
    resulting_list[missing_num] = min_val

    transfer_list[3] = resulting_list

    return transfer_list

def update_variables():
    global n_vector, n_transfer,neuronsPerNode, neuronsPerCore, spikesPerNodePerTimestep, synapses_per_Node, bytes_per_synaptic_list

    network_size = list(network_size_dict.values())
    num_Nodes       =   np.prod(network_size)
    num_Neighbours_per_Node = sum([x-1 for x in network_size])
    num_indirect_Neighbours_per_Node = sum([np.prod(i) for i in combinations([x-1 for x in network_size], 2)])
    num_2nd_indirect_Neighbours_per_Node = sum([np.prod(i) for i in combinations([x-1 for x in network_size], 3)])
    num_3rd_indirect_Neighbours_per_Node = sum([np.prod(i) for i in combinations([x-1 for x in network_size], 4)])
    n_vector = [1, num_Neighbours_per_Node, num_indirect_Neighbours_per_Node, num_2nd_indirect_Neighbours_per_Node, num_3rd_indirect_Neighbours_per_Node]
    n_transfer = calculate_transfer_list(network_size)

    neuronsPerNode  =   bio_constants['Neurons'] / num_Nodes
    neuronsPerCore  =   neuronsPerNode / hardware_settings['Number of Cores']
    spikesPerNodePerTimestep = neuronsPerNode * bio_constants['Fire rate']*bio_constants['Time resolution']
    synapses_per_Node = bio_constants['Synapses per Neuron'] / num_Nodes
    bytes_per_synaptic_list	= math.ceil(synapses_per_Node * hardware_settings['Bytes per synapse'] / (hardware_settings['DRAM fetch size'] / 8)) * (hardware_settings['DRAM fetch size'] / 8)

    info_dataframe.at['DRAM to Spike Dispatcher', 'Data Size'] = unit_element(str(bytes_per_synaptic_list)+" Byte")
    info_dataframe.at['DRAM to Spike Dispatcher', 'max. rel. usable Data_Rate'] = calculate_dram_data_rate()

    if print_info != "none":
        print_settings()

def save_settings(file_path):
    global current_file_path

    current_file_path = file_path
    file = open(file_path, "w")

    file.write("bio_constants\n")
    for identifier, value in bio_constants.items():
        file.write("{: <25} =   {}\n".format(identifier, value))

    file.write("\nhardware_settings\n")
    for identifier, value in hardware_settings.items():
        file.write("{: <25} =   {}\n".format(identifier, value))

    file.write("\nnetwork_size\n")
    for identifier, value in network_size_dict.items():
        file.write("{: <25} =   {}\n".format(identifier, value))

    file.write("\nlatency\n")
    for identifier, value in latency.items():
        file.write("{: <25} =   {}\n".format(identifier, value))

    exclude_columns = ['max. rel. usable Data_Rate', 'affect', 'visualize', 'color']
    df_subset = info_dataframe.drop(columns=exclude_columns)

    file.write("\nstages")

    for index, stage in info_dataframe.iterrows():
        file.write("\n{: <25} =   {}\n".format("index", index))
        for column in df_subset.columns:
            connected_setting = [index, column]
            if connected_setting == ['DRAM to Spike Dispatcher','Data Size']:
                file.write("{: <25} =   {}\n".format(column, "bytes_per_synaptic_list Byte"))
                continue

            for sublist in connected_settings:
                if connected_setting in sublist:
                    file.write("{: <25} =   {}\n".format(column, sublist[0] + " " + stage[column].unit))
                    break
            else:
                file.write("{: <25} =   {}\n".format(column, stage[column]))

    file.write("\ncolors\n")
    print(color_dict)
    for identifier, value in color_dict.items():
        file.write("{: <25} =   {}\n".format(identifier, value))
    file.close()

def print_settings():
    print("\nBiological Constants:")
    for index, elem in bio_constants.items():
        print("\t{: <25} {: <20}".format(*(index, elem)))
    print()

    print("Hardware Settings:")
    for index, elem in network_size_dict.items():
        print("\t{: <25} {: <20}".format(*(index, elem)))
    print()
    for index, elem in hardware_settings.items():
        print("\t{: <25} {: <20}".format(*(index, elem)))
    print()

    if print_info == "detailed":
        print("Stages information:")
        info_df_configured_elements = info_dataframe[['Task',
                                           'Number of Channel',
                                           'Max. Data Rate',
                                           'Data Size',
                                           'depend',
                                           'Added Latency',
                                           'Hardwareclass']]
        for index, elem in info_df_configured_elements.iterrows():
            print(f"\t{index}")
            for column in info_df_configured_elements.columns:
                print("\t{: <25} {: <20}".format(*(column, elem[column])))
            print()