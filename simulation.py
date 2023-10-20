#-----------------------------------------------------------------------------
# File Name : simulation.py
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

import settings

class Simulation:
    def __init__(self):
        self.dimension = np.count_nonzero(np.array([settings.network_size_dict['Network Height'], settings.network_size_dict['Network Width'], settings.network_size_dict['Network Depth'], settings.network_size_dict['Network 4Dim']]) - np.array([1,1,1,1]))
        self.create_dataframe()

        self.calculaute_used_bandwidth()
        self.generate_latency_dataframe()
        self.generate_duration_extension_dict()

        self.total_latency = max(self.timing_dataframe['End'])
        self.acceleration_factor = 1/self.total_latency*1e9*settings.bio_constants['Time resolution']

        if settings.print_info != "none":
            self.print_results()

    ##
    # function creates a timing_dataframe which only considers the latencies
    # and for the stage 'Write into RingBuffer' a duration extension to read a read request from the DRAM.
    def create_dataframe(self):
        start = 0
        duration = (math.ceil(settings.neuronsPerCore)+1)/settings.hardware_settings['Neuron clock']
        self.timing_dataframe =  pd.DataFrame({'Task':['Neuron calculation'],
                                    'Start':start,
                                    'End':start + duration,
                                    'Duration': duration,
                                    'Hardwareclass': ['Calculation']})
        for i in range(self.dimension+1):
            for index, data_type in settings.info_dataframe.iterrows():
                if (i != 0 or index != 'Router to Spike Dispatcher') and (i != self.dimension or index != 'Interconnect'):
                    if i == 0 and (index == 'Interconnect' or index == 'Router to DRAM'):
                        start = settings.hardware_settings['Pipeline depth']/settings.hardware_settings['Neuron clock'] + settings.latency['CDC'] + settings.latency['Router']
                    else:
                        start = self.timing_dataframe.loc[self.timing_dataframe.Task==data_type['depend'], 'Start'].values.tolist()[-1] + data_type['Added Latency']
                        duration = self.timing_dataframe.loc[self.timing_dataframe.Task==data_type['depend'], 'Duration'].values.tolist()[-1]
                    
                    if data_type['Task'] == 'Write into RingBuffer':
                        delta_time = data_type['Data Size']/data_type['Max. Data Rate']
                        duration = self.timing_dataframe.loc[self.timing_dataframe.Task==data_type['depend'], 'Duration'].values.tolist()[-1] + delta_time

                    temp_df = pd.DataFrame({'Task': [data_type['Task']],
                                        'Start':start,
                                        'End':start + duration,
                                        'Duration': duration,
                                        'Hardwareclass': [data_type['Hardwareclass']]})
                    
                    self.timing_dataframe = pd.concat([self.timing_dataframe, temp_df], ignore_index=True)
        self.max_timing_df_index = max(self.timing_dataframe.index)

    ##
    # function calculates the necessary data rate for all stages
    # and, if the data rate exceeds the maximum data rate, calls update_used_bandwidth
    def calculaute_used_bandwidth(self):
        self.ideal_rel_data_rate = {}
        self.ideal_uncombined_intervals = {}
        self.final_rel_data_rate = {}
        self.final_intervals = {}
        for task, current_data_size, current_num_of_channel, current_max_channel_data_rate, current_max_rel_usable_data_rate in (settings.info_dataframe.loc[:, ['Task', 'Data Size', 'Number of Channel', 'Max. Data Rate', 'max. rel. usable Data_Rate']].values.tolist()):
            timing_df_entries_with_same_task, rel_data_rate = self.get_rel_data_rate(task, current_num_of_channel, current_data_size, current_max_channel_data_rate)
            uncombined_intervals = timing_df_entries_with_same_task[['Start', 'End']].values

            overlaps = self.check_overlap(uncombined_intervals)
            if len(overlaps) > 0:
                new_intervals, new_data_rate, overlaps_finished = self.compose_uncombined_intervals(overlaps,uncombined_intervals, rel_data_rate, task, float(current_max_rel_usable_data_rate), True)
            else:
                new_intervals = uncombined_intervals
                new_data_rate = rel_data_rate

            if any(element > current_max_rel_usable_data_rate for element in new_data_rate):
                indexes = [index for index, elem in enumerate(new_data_rate) if elem < current_max_rel_usable_data_rate]

                self.ideal_rel_data_rate[task] = np.delete(new_data_rate, indexes)
                self.ideal_uncombined_intervals[task] = np.delete(new_intervals, indexes, 0)
                self.check_max_data_rate(rel_data_rate, timing_df_entries_with_same_task, task, float(current_max_rel_usable_data_rate))
                self.increase_dependent_duration()
                self.extend_following_duration_to_minimum()
                self.update_used_bandwidth(task)
            else:
                self.ideal_rel_data_rate[task] = []
                self.ideal_uncombined_intervals[task] = []
                self.final_rel_data_rate[task] = new_data_rate
                self.final_intervals[task] = new_intervals

    ##
    # function updates the data rate for a task if the previously calculated data rates were greater than the maximum possible
    def update_used_bandwidth(self, task):
        current_data_size, current_num_of_channel, current_max_channel_data_rate, current_max_rel_usable_data_rate = settings.info_dataframe.loc[settings.info_dataframe.Task == task, ['Data Size', 'Number of Channel', 'Max. Data Rate', 'max. rel. usable Data_Rate']].values[0]
        while True:
            timing_df_entries_with_same_task, rel_data_rate = self.get_rel_data_rate(task, current_num_of_channel, current_data_size, current_max_channel_data_rate)
            uncombined_intervals = timing_df_entries_with_same_task[['Start', 'End']].values
            overlaps = self.check_overlap(uncombined_intervals)
            if len(overlaps) > 0:
                final_intervals, final_rel_data_rate, overlaps_finished = self.compose_uncombined_intervals(overlaps,uncombined_intervals, rel_data_rate, task, float(current_max_rel_usable_data_rate), False)
                if overlaps_finished:
                    self.final_rel_data_rate[task] = final_rel_data_rate
                    self.final_intervals[task] = final_intervals
                    break
                self.extend_following_duration_to_minimum()
            else:
                self.final_intervals[task] = uncombined_intervals
                self.final_rel_data_rate[task] = rel_data_rate
                break
        self.increase_dependent_duration()

    ##
    # function calculates the non-overlapping data rates of a stage
    # and returns the data rates and as well as the timing_dataframe entries
    def get_rel_data_rate(self, task, current_num_of_channel, current_data_size, current_max_channel_data_rate):
        timing_df_entries_with_same_task = self.timing_dataframe.loc[self.timing_dataframe.Task==task].reset_index(drop=True)
        if task == 'Transmitting Spikes':
            n_vector = [max(i) for i in settings.n_transfer if i != [] ]
        else:
            n_vector = settings.n_vector[(self.dimension+1-len(timing_df_entries_with_same_task)):(self.dimension+1)]

        rel_data_rate = np.array(np.divide(n_vector, timing_df_entries_with_same_task["Duration"])) * settings.spikesPerNodePerTimestep / int(current_num_of_channel) * (current_data_size/current_max_channel_data_rate)
        return timing_df_entries_with_same_task, rel_data_rate

    ## function checks if a stage is terminated after the following ones and
    # extends the duration of the following ones if necessary, so that this is no longer the case.
    def extend_following_duration_to_minimum(self):
        for index, row in settings.info_dataframe[['Task', 'affect']].iterrows():
            if np.atleast_1d(row['affect']) is None:
                for index, i in enumerate(self.timing_dataframe.loc[self.timing_dataframe.Task==row['Task']].index[:-1]):
                    if self.timing_dataframe.at[i, 'End'] > self.timing_dataframe.at[self.timing_dataframe.loc[self.timing_dataframe.Task==row['Task']].index[index+1], 'End']:
                        self.timing_dataframe.at[self.timing_dataframe.loc[self.timing_dataframe.Task==row['Task']].index[index+1], 'End'] = self.timing_dataframe.at[i, 'End'] + 1
                        self.timing_dataframe.at[self.timing_dataframe.loc[self.timing_dataframe.Task==row['Task']].index[index+1], 'Duration'] = self.timing_dataframe.at[self.timing_dataframe.loc[self.timing_dataframe.Task==row['Task']].index[index+1], 'End'] - self.timing_dataframe.at[self.timing_dataframe.loc[self.timing_dataframe.Task==row['Task']].index[index+1], 'Start']

    ##
    # function extends the duration of a subsequent Task if it is less
    def increase_dependent_duration(self):
        to_update_data_rates = []
        for timing_df_index in self.timing_dataframe.index:# [i for i in self.timing_dataframe.index if i > 0]:
            for affected_tasks in (settings.info_dataframe.loc[settings.info_dataframe.Task==self.timing_dataframe.at[timing_df_index, 'Task']]['affect']).values.tolist():
                for affected_task in np.atleast_1d(affected_tasks):
                    following_dep_indexes = [i for i in self.timing_dataframe.loc[self.timing_dataframe.Task==affected_task].index if i > timing_df_index]
                    if len(following_dep_indexes) == 0:
                        continue

                    following_dep_index = min(following_dep_indexes)
                    if affected_task == 'Write into RingBuffer':
                        info_dataframe_entry = settings.info_dataframe.loc[settings.info_dataframe.Task == affected_task]
                        delta_duration = info_dataframe_entry['Data Size'][0]/info_dataframe_entry['Max. Data Rate'][0]
                    else:
                        delta_duration = 0

                    if self.timing_dataframe.at[timing_df_index, 'Duration'] + delta_duration > self.timing_dataframe.at[following_dep_index, 'Duration']:
                        self.timing_dataframe.at[following_dep_index, 'Duration'] = self.timing_dataframe.at[timing_df_index, 'Duration'] + delta_duration
                        self.timing_dataframe.at[following_dep_index, 'End'] = self.timing_dataframe.at[following_dep_index, 'Start'] + self.timing_dataframe.at[following_dep_index, 'Duration']

                        if affected_task in self.final_intervals.keys():
                            to_update_data_rates.append(affected_task)
        for task in to_update_data_rates:
            self.update_used_bandwidth(task)

    ##
    # function merges the uncombined intervals and data rates and
    # returns a boolean value indicating whether duration extension occurred during the operation,
    # and the final data rates and intervals
    def compose_uncombined_intervals(self, overlaps, uncombined_intervals, rel_data_rate, task, current_max_rel_usable_data_rate, init):
        for i in range(overlaps[0][0]+1):
            if i == 0:
                final_intervals = [uncombined_intervals[i]]
                final_rel_data_rate = [rel_data_rate[i]]
            else:
                final_intervals = np.insert(final_intervals, i, [uncombined_intervals[i]], axis=0)
                final_rel_data_rate = np.insert(final_rel_data_rate, i, [rel_data_rate[i]], axis=0)
        max_num = 1
        overlaps_finished_final = True
        for index in range(len(overlaps)):
            if overlaps[index][1]-overlaps[index][0] == 1 and (index == 0 or ((overlaps[index-1][1]-overlaps[index-1][0]) == 1)):
                final_intervals, final_rel_data_rate, overlaps_finished = self.insert_overlap(index=index,
                                              task=task,
                                              uncombined_intervals=uncombined_intervals,
                                              overlaps=overlaps,
                                              distance_of_overlapping_elements=1,
                                              rel_data_rate=rel_data_rate,
                                              current_max_rel_usable_data_rate=current_max_rel_usable_data_rate,
                                              final_intervals=final_intervals,
                                              final_rel_data_rate=final_rel_data_rate,
                                              init=init)

            elif overlaps[index][1]-overlaps[index][0] >= max_num:
                max_num = overlaps[index][1]-overlaps[index][0]
                final_intervals, final_rel_data_rate, overlaps_finished = self.insert_overlap(index=index,
                                              task=task,
                                              uncombined_intervals=uncombined_intervals,
                                              overlaps=overlaps,
                                              distance_of_overlapping_elements=max_num,
                                              rel_data_rate=rel_data_rate,
                                              current_max_rel_usable_data_rate=current_max_rel_usable_data_rate,
                                              final_intervals=final_intervals,
                                              final_rel_data_rate=final_rel_data_rate,
                                              init=init)

            if overlaps_finished is False:
                if not init:
                    return final_intervals, final_rel_data_rate, overlaps_finished
                else:
                    overlaps_finished_final = False
        return final_intervals, final_rel_data_rate, overlaps_finished_final

    def insert_overlap(self, index, task, uncombined_intervals, overlaps, distance_of_overlapping_elements, rel_data_rate, current_max_rel_usable_data_rate, final_intervals, final_rel_data_rate, init):
        overlap_finished = True
        final_intervals = self.combine_intervals(uncombined_intervals=uncombined_intervals,
                                    highest_overlapping_element=overlaps[index][1],
                                            overlap_start=overlaps[index][2],
                                            overlap_end=overlaps[index][3],
                                            distance_of_overlapping_elements=distance_of_overlapping_elements,
                                            final_intervals=final_intervals)

        overlapping_rate = final_rel_data_rate[-distance_of_overlapping_elements] + rel_data_rate[overlaps[index][1]]
        final_rel_data_rate = self.insert_data_rate(highest_overlapping_element=overlaps[index][1],
                                               distance_of_overlapping_elements=distance_of_overlapping_elements,
                                               overlapping_rate=overlapping_rate,
                                               rel_data_rate=rel_data_rate,
                                               final_rel_data_rate=final_rel_data_rate)

        if overlapping_rate - current_max_rel_usable_data_rate > settings.accuracy:
            data_volume = self.get_data_overhead(highest_overlapping_element=overlaps[index][1],
                                                                                        distance_of_overlapping_elements=distance_of_overlapping_elements,
                                                                                        overlapping_rate=overlapping_rate,
                                                                                        current_max_rel_usable_data_rate=current_max_rel_usable_data_rate,
                                                                                        overlap_start=overlaps[index][2],
                                                                                        overlap_end=overlaps[index][3],
                                                                                        rel_data_rate=rel_data_rate,
                                                                                        final_rel_data_rate=final_rel_data_rate,
                                                                                        final_intervals=final_intervals)

            data_volume, final_rel_data_rate = self.redistribute_data_rate(highest_overlapping_element=overlaps[index][1],
                                                    distance_of_overlapping_elements=distance_of_overlapping_elements,
                                                    current_max_rel_usable_data_rate=current_max_rel_usable_data_rate,
                                                    final_intervals=final_intervals,
                                                    final_rel_data_rate=final_rel_data_rate,
                                                    data_volume=data_volume)

            if data_volume > settings.accuracy:
                if not init:
                    self.extend_duration(task=task,
                                        data_volume=data_volume,
                                        current_max_rel_usable_data_rate=current_max_rel_usable_data_rate,
                                        highest_overlapping_element=overlaps[index][1])
                    return final_intervals, final_rel_data_rate, False
                else:
                    final_rel_data_rate = self.increase_ideal_data_rate(distance_of_overlapping_elements=distance_of_overlapping_elements,
                                                  final_intervals=final_intervals,
                                                  final_rel_data_rate=final_rel_data_rate,
                                                  data_volume=data_volume)
                    overlap_finished = False
        return final_intervals, final_rel_data_rate, overlap_finished

    def combine_intervals(self, uncombined_intervals, highest_overlapping_element, overlap_start, overlap_end, distance_of_overlapping_elements, final_intervals):
        final_intervals[-distance_of_overlapping_elements][1] = overlap_start
        new_interval = np.array([[overlap_start, overlap_end]])
        final_intervals = np.insert(final_intervals, len(final_intervals)-(distance_of_overlapping_elements-1), [new_interval], axis=0)
        new_interval = np.array([[final_intervals[-1][1], uncombined_intervals[highest_overlapping_element][1]]])
        final_intervals = np.insert(final_intervals, len(final_intervals), [new_interval], axis=0)
        return final_intervals

    ##
    # function inserts the data rate so that it is added to all overlapping data rates
    def insert_data_rate(self, highest_overlapping_element, distance_of_overlapping_elements, overlapping_rate, rel_data_rate, final_rel_data_rate):
        final_rel_data_rate = np.insert(final_rel_data_rate, len(final_rel_data_rate)-(distance_of_overlapping_elements-1), [overlapping_rate], axis=0)
        for i in range(1,distance_of_overlapping_elements):
            final_rel_data_rate[-i] += rel_data_rate[highest_overlapping_element]
        final_rel_data_rate = np.insert(final_rel_data_rate, len(final_rel_data_rate), [rel_data_rate[highest_overlapping_element]], axis=0)
        return final_rel_data_rate

    def get_data_overhead(self, highest_overlapping_element, distance_of_overlapping_elements, overlapping_rate, current_max_rel_usable_data_rate, overlap_start, overlap_end, rel_data_rate, final_rel_data_rate, final_intervals):
        data_volume = 0
        for i in range(1,(distance_of_overlapping_elements+2)):
            if(final_rel_data_rate[-i] > current_max_rel_usable_data_rate):
                data_volume += (final_rel_data_rate[-i] - current_max_rel_usable_data_rate) * (final_intervals[-i][1]-final_intervals[-i][0])
        return data_volume

    ##
    # function distributes the excess data volume on available_data_overhead
    # and returns the resulting data rate and the data volume that could not be distributed
    def redistribute_data_rate(self, highest_overlapping_element, distance_of_overlapping_elements, current_max_rel_usable_data_rate, final_intervals, final_rel_data_rate, data_volume):
        available_data_overhead = 0

        for i in range(1,(distance_of_overlapping_elements+2)):
            if final_rel_data_rate[-i] > current_max_rel_usable_data_rate:
                final_rel_data_rate[-i] = current_max_rel_usable_data_rate

        for i in range(1,(distance_of_overlapping_elements+1)):
            delta_interval = final_intervals[-1][1]-final_intervals[-i][0]
            available_data_overhead = (final_rel_data_rate[-(i+1)] - final_rel_data_rate[-i]) * delta_interval
            if data_volume - available_data_overhead <= settings.accuracy:
                for j in range(1, (i+1)):
                    final_rel_data_rate[-j] += data_volume / delta_interval
                data_volume = 0
            else:
                for j in range(1, (i+1)):
                    final_rel_data_rate[-j] = final_rel_data_rate[-(i+1)]
                data_volume -= available_data_overhead

        return data_volume, final_rel_data_rate

    ##
    # function extends the duration by the part that the remaining data volume can be transferred
    def extend_duration(self, task, data_volume, current_max_rel_usable_data_rate, highest_overlapping_element):
        delta_duration = data_volume / current_max_rel_usable_data_rate
        timing_df_index = self.timing_dataframe.loc[self.timing_dataframe.Task==task].index[highest_overlapping_element]
        temp_entry = self.timing_dataframe.at[timing_df_index, 'Duration']
        self.timing_dataframe.at[timing_df_index, 'Duration'] = temp_entry + delta_duration
        self.timing_dataframe.at[timing_df_index, 'End'] = self.timing_dataframe.at[timing_df_index, 'Start'] + temp_entry + delta_duration

    ##
    # function determines the data rate that would be necessary to transfer the data volume without duration extension
    def increase_ideal_data_rate(self, distance_of_overlapping_elements, final_intervals, final_rel_data_rate, data_volume):
        delta_interval = final_intervals[-1][1]-final_intervals[-(distance_of_overlapping_elements+1)][0]
        for i in range(1,(distance_of_overlapping_elements+2)):
                final_rel_data_rate[-i] += data_volume / delta_interval
        return final_rel_data_rate

    ##
    # function extends the duration if the uncombined data rate is higher than the maximum data rate
    def check_max_data_rate(self, rel_data_rate, timing_df_entries_with_same_task, task, max_rate):
        for index, i in enumerate(self.timing_dataframe.loc[self.timing_dataframe.Task==task].index):
            if rel_data_rate[index] > max_rate:
                self.timing_dataframe.at[i, 'Duration'] = rel_data_rate[index]/max_rate * timing_df_entries_with_same_task["Duration"][index]
                self.timing_dataframe.at[i, 'End'] = self.timing_dataframe.at[i, 'Start'] + self.timing_dataframe.at[i, 'Duration']

    ##
    # Function determines which intervals overlap in a stage
    # and returns a list containing the indexes, start and end times of the overlapping elements
    def check_overlap(self, pairs):
        overlaps = []
        for i in range(len(pairs)):
            for j in range(i+1, len(pairs)):
                pair1 = pairs[i]
                pair2 = pairs[j]
                if pair1[1] >= pair2[0] and pair1[0] <= pair2[1]:
                    overlap_start = max(pair1[0], pair2[0])
                    overlap_end = min(pair1[1], pair2[1])
                    overlaps.append((i, j, overlap_start, overlap_end))
        return overlaps

    ##
    # The function determines which latency elements to plot so that no latencies are plotted twice
    # and returns them
    def get_latency_elements_to_plot(self, info_dataframe_index, data_type, index):
        current_latency_elements = settings.info_dataframe.at[info_dataframe_index, 'Added Latency'].elements.copy()

        tasks_with_same_dependecy = [settings.info_dataframe.at[key, 'Task'] for key,val in settings.task_dependency.items() if val == settings.info_dataframe.at[info_dataframe_index, 'depend'] and key != info_dataframe_index]

        previous_latency_element = []
        if tasks_with_same_dependecy != []:
            for task_with_same_dependecy in tasks_with_same_dependecy:
                task_with_same_dependecy_info_df_index = settings.info_dataframe.index[settings.info_dataframe.Task == task_with_same_dependecy][0]

                current_task_time_df_indexes = self.timing_dataframe.loc[self.timing_dataframe.Task == data_type['Task']].index.values.tolist()
                previous_time_df_index_of_current_task = max([i for i in current_task_time_df_indexes if i < index] + [0])

                task_with_same_dependecy_time_df_indexes = self.timing_dataframe.loc[self.timing_dataframe.Task == task_with_same_dependecy].index.values.tolist()
                previous_time_df_index_of_task_with_same_dependecy = [i for i in task_with_same_dependecy_time_df_indexes if i < index and i > previous_time_df_index_of_current_task]
                if settings.info_dataframe.index.get_loc(task_with_same_dependecy_info_df_index) < settings.info_dataframe.index.get_loc(info_dataframe_index) and previous_time_df_index_of_task_with_same_dependecy != []:
                    previous_latency_element = previous_latency_element + settings.info_dataframe.at[task_with_same_dependecy_info_df_index, 'Added Latency'].elements

        for elem in previous_latency_element:
            if len(current_latency_elements) > 0 and elem in current_latency_elements:
                current_latency_elements.remove(elem)
        return current_latency_elements

    def generate_latency_dataframe(self):
        start = 0
        duration = settings.hardware_settings['Pipeline depth']/settings.hardware_settings['Neuron clock']
        self.latency_dataframe =  pd.DataFrame({'Position': max(self.timing_dataframe.index)-1,
                                                'Start':start,
                                                'End':start + duration,
                                                'Duration': duration,
                                                'Hardwareclass': ['Pipeline']})
        temp_df = pd.DataFrame({'Position': max(self.timing_dataframe.index)-1,
                                'Start':start + duration,
                                'End':start + duration + settings.latency['CDC'],
                                'Duration': settings.latency['CDC'],
                                'Hardwareclass': ['CDC']})

        self.latency_dataframe = pd.concat([self.latency_dataframe, temp_df], ignore_index=True)

        start = start + duration + settings.latency['CDC']
        duration = settings.latency['Router']
        temp_df = pd.DataFrame({'Position': max(self.timing_dataframe.index)-1,
                                'Start':start,
                                'End':start + duration,
                                'Duration': duration,
                                'Hardwareclass': ['Router']})

        self.latency_dataframe = pd.concat([self.latency_dataframe, temp_df], ignore_index=True)

        for index, data_type in self.timing_dataframe.loc[self.timing_dataframe.index > 0].iterrows():
            if index < len(settings.info_dataframe) and (data_type['Task'] == 'Requesting Lookup' or data_type['Task'] == 'Transmitting Spikes'):
                continue
            info_dataframe_index = settings.info_dataframe.index[settings.info_dataframe.Task == data_type['Task']][0]

            current_latency_elements = self.get_latency_elements_to_plot(info_dataframe_index, data_type, index)

            current_end_point = data_type['Start']
            pos = max(self.timing_dataframe.index)-int(index)
            for latency_element in current_latency_elements[::-1]:
                if latency_element != '':
                    duration = settings.latency[latency_element]
                    if latency_element == 'Additional Hop Delay':
                        duration = sum([i-2 for i in list(settings.network_size_dict.values()) if i > 2]) * float(settings.latency[latency_element]) / (3+3)
                    start = current_end_point - duration
                    temp_df = pd.DataFrame({'Position': pos,
                                        'Start':start,
                                        'End':current_end_point,
                                        'Duration': duration,
                                        'Hardwareclass': [latency_element]})
                    self.latency_dataframe = pd.concat([self.latency_dataframe, temp_df], ignore_index=True)
                    current_end_point = start

    def generate_duration_extension_dict(self):
        self.extension_dict = {}

        for index, data_type in self.timing_dataframe.loc[self.timing_dataframe.index > 0].iterrows():
            depending_task = settings.info_dataframe.loc[settings.info_dataframe.Task == data_type['Task'], 'depend'].values.tolist()[0]
            depending_indexes = self.timing_dataframe.loc[self.timing_dataframe.Task == depending_task].index

            if len(depending_indexes) == 0 or min(depending_indexes) > index:
                ref_duration = self.timing_dataframe.at[0,'Duration']
            else:
                depending_index = max([i for i in self.timing_dataframe.loc[self.timing_dataframe.Task == depending_task].index if i < index])
                ref_duration = self.timing_dataframe.at[depending_index,'Duration']

            if data_type['Task'] == 'Write into RingBuffer':
                info_dataframe_entry = settings.info_dataframe.loc[settings.info_dataframe.Task == data_type['Task']]
                ref_duration += info_dataframe_entry['Data Size'][0]/info_dataframe_entry['Max. Data Rate'][0]
            if ref_duration < data_type['Duration']:
                duration_extension = data_type['Duration'] - ref_duration

                self.extension_dict[index] = (ref_duration, duration_extension)

    def print_results(self):
        if settings.print_info == "detailed":
            print("Timing Dataframe:")
            for time_df_index, time_df_row in self.timing_dataframe[['Task', 'Start', 'End', 'Duration']].iterrows():
                print("\tLatencies")
                for latency_df_index, latency_df_row in self.latency_dataframe.loc[self.latency_dataframe.Position==(self.max_timing_df_index-time_df_index), ['Hardwareclass', 'Start', 'End', 'Duration']].iterrows():
                    print(f"\t{latency_df_row['Hardwareclass']} – Latency")
                    for latency_df_column, entry_value in latency_df_row[1:].items():
                        print("\t{: <25} {: <20}".format(*(latency_df_column, f"{round(entry_value,2)} ns")))
                    print()

                print("\t{: <25} {: <20}".format(*("Task", time_df_row['Task'])))

                for time_df_column, entry_value in time_df_row[1:].items():
                    print("\t{: <25} {: <20}".format(*(time_df_column, f"{round(entry_value,2)} ns")))

                if time_df_index in self.extension_dict.keys():
                    max_data_rate = settings.info_dataframe.loc[settings.info_dataframe.Task == time_df_row['Task'], 'Max. Data Rate'][0]
                    rel_max_data_rate = settings.info_dataframe.loc[settings.info_dataframe.Task == time_df_row['Task'], 'max. rel. usable Data_Rate'][0]
                    max_required_data_rate = max(self.ideal_rel_data_rate[time_df_row['Task']]) / rel_max_data_rate * max_data_rate
                    print("\t!!Duration extension due to reaching the maximum Data rate!!\n",
                          "\tmax required Data rate:", max_required_data_rate.round(2))
                print("\n")

        print(f"Acceleration Factor: {round(self.acceleration_factor,3)}")
        print(f"Total Latency: {round(self.total_latency,3)}")