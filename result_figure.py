import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import textwrap
import os

import settings
import simulation

class Result_figure():
    def __init__(self):
        self.padding_left = settings.padding_left
        self.padding_right = settings.padding_right
        self.padding_top = settings.padding_top
        self.padding_bottom = settings.padding_bottom
        self.bbox_padding_x = settings.bbox_padding
        self.bbox_padding_y = settings.bbox_padding
        self.bbox_width = settings.bbox_width
        self.name_bar_padding = settings.name_bar_padding

        if settings.no_gui:
            plt.ioff()

        self.fig = plt.figure(dpi=200)

        self.ax_text_tasks = {}
        self.ax_text_latencies = {}

        self.initialize_figure()

    def initialize_figure(self):
        self.ax_data_rate = {}
        self.ax_ylabel_boxes = {}
        self.ax_ylabel_labels = {}

        element_list = settings.info_dataframe.loc[:, 'Task'].values.tolist()
        element_list = ['Gantt'] + element_list
        height_ratio = [4] + [1] * (len(element_list)-1)

        for i in range(5):
            self.ax_ylabel_boxes[i] = patches.FancyBboxPatch((0.5, 0.5),
                                                                width = 0.2,
                                                                height = 0.2,
                                                                ec="k",
                                                                fc = settings.node_box_background_color,
                                                                transform=self.fig.transFigure,
                                                                boxstyle=patches.BoxStyle("Round", pad=self.bbox_padding_x))
            self.fig.add_artist(self.ax_ylabel_boxes[i])
            self.ax_ylabel_boxes[i].set_visible(False)
            self.ax_ylabel_labels[i] = self.fig.text(0.5, 0.5, "Node N" if i == 0 else f"Node\nN+{i}",
                                                        transform=self.fig.transFigure,
                                                        ha='center', va='center', fontsize=8)
            self.ax_ylabel_labels[i].set_visible(False)

        for info_df_index, info_df_entry in settings.info_dataframe.iterrows():
            self.ax_ylabel_boxes[info_df_entry['Task']] = patches.FancyBboxPatch((0.5, 0.5),
                                                                width = 0.2,
                                                                height = 0.2,
                                                                ec="k",
                                                                fc = settings.color_dict[info_df_entry['Hardwareclass']],
                                                                transform=self.fig.transFigure,
                                                                boxstyle=patches.BoxStyle("Round", pad=self.bbox_padding_x))
            self.fig.add_artist(self.ax_ylabel_boxes[info_df_entry['Task']])
            self.ax_ylabel_labels[info_df_entry['Task']] = self.fig.text(1, 1, "",
                                                        transform=self.fig.transFigure,
                                                        ha='center', va='center', fontsize=5)
            if not info_df_entry['visualize']:
                self.ax_ylabel_boxes[info_df_entry['Task']].set_visible(False)
                self.ax_ylabel_labels[info_df_entry['Task']].set_visible(False)

        axs = self.fig.subplots(nrows=len(element_list), sharex=True, gridspec_kw={'height_ratios': height_ratio})
        if type(axs).__module__ == np.__name__:
            for task,ax in zip(element_list, axs.ravel()):
                if task == 'Gantt':
                    self.ax_figure = ax
                    ax.xaxis.tick_top()
                    ax.tick_params(axis='x', labeltop=True)
                else:    
                    self.ax_data_rate[task] = ax
        else:
            self.ax_figure = axs

        self.reset_figure()

    def reset_figure(self):
        for ax in self.ax_data_rate.values():
            ax.clear()
        self.ax_figure.clear()

        self.ax_figure.set_yticks([])
        self.ax_figure.set_facecolor(settings.background_color)
        self.ax_figure.spines['right'].set_visible(False)
        self.ax_figure.spines['left'].set_visible(False)
        self.ax_figure.spines['left'].set_position(('outward', 10))
        self.ax_figure.spines['top'].set_visible(False)
        self.ax_figure.spines['bottom'].set_color('gray')
        self.ax_figure.grid(True, lw=0.1)

        for index, ax_data_rate in self.ax_data_rate.items():
            ax_data_rate.tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
            ax_data_rate.set_yticks([])
            ax_data_rate.set_ylim([-0.1, 1.1])
            ax_data_rate.set_facecolor(settings.background_color)
            ax_data_rate.spines['right'].set_visible(False)
            ax_data_rate.spines['left'].set_visible(False)
            ax_data_rate.spines['left'].set_position(('outward', 10))
            ax_data_rate.spines['top'].set_visible(False)
            ax_data_rate.spines['bottom'].set_color('gray')
            ax_data_rate.grid(True, lw=0.1)

        for i in range(5):
            self.ax_ylabel_boxes[i].set_visible(False)
            self.ax_ylabel_labels[i].set_visible(False)

        self.update_axes()
        self.reset_name_bars()
        plt.draw()

    ##
    # function updates the size and position of the axes and the ax_ylabel_boxes next to them
    def update_axes(self):
        for i, info_df_entry_not_visualized in settings.info_dataframe.loc[settings.info_dataframe.visualize==False, ['Task', 'visualize']].reset_index(drop=True).iterrows():
            self.ax_data_rate[info_df_entry_not_visualized['Task']].set_visible(False)
            self.ax_ylabel_boxes[info_df_entry_not_visualized['Task']].set_visible(False)
            self.ax_ylabel_labels[info_df_entry_not_visualized['Task']].set_visible(False)

        visualized_info_dataframe = settings.info_dataframe.loc[settings.info_dataframe.visualize==True, ['Task', 'visualize']].reset_index(drop=True)
        height_ratio = [4] + [1] * len(visualized_info_dataframe)
        self.ax_figure.set_position([self.padding_left, self.padding_bottom + sum(height_ratio[1:]) / sum(height_ratio) * (self.padding_top-self.padding_bottom),
                                     (self.padding_right-self.padding_left), height_ratio[0] / sum(height_ratio) * (self.padding_top-self.padding_bottom)])

        bbox_height = height_ratio[-1] / sum(height_ratio) * (self.padding_top-self.padding_bottom)

        for i, visualized_info_df_entry in visualized_info_dataframe.iterrows():
            if not visualized_info_df_entry['visualize']:
                self.ax_data_rate[visualized_info_df_entry['Task']].set_visible(False)
                self.ax_ylabel_boxes[visualized_info_df_entry['Task']].set_visible(False)
                self.ax_ylabel_labels[visualized_info_df_entry['Task']].set_visible(False)
                continue

            self.ax_data_rate[visualized_info_df_entry['Task']].set_position([self.padding_left, self.padding_bottom + sum(height_ratio[(i+2):]) / sum(height_ratio) * (self.padding_top-self.padding_bottom),
                                                                              (self.padding_right-self.padding_left),
                                                                              height_ratio[(i+1)] / sum(height_ratio) * (self.padding_top-self.padding_bottom)])

            visible_axes = [key for key, ax in self.ax_data_rate.items() if ax.get_visible()]
            index_b = visible_axes.index(visualized_info_df_entry['Task'])
            bbox_y = self.padding_bottom + sum(height_ratio[(index_b+2):]) / sum(height_ratio) * (self.padding_top-self.padding_bottom) + self.bbox_padding_x

            self.ax_ylabel_boxes[visualized_info_df_entry['Task']].set_x(self.padding_left - self.bbox_width - self.bbox_padding_x)
            self.ax_ylabel_boxes[visualized_info_df_entry['Task']].set_y(bbox_y)
            self.ax_ylabel_boxes[visualized_info_df_entry['Task']].set_width(self.bbox_width)
            self.ax_ylabel_boxes[visualized_info_df_entry['Task']].set_height(bbox_height-2*self.bbox_padding_y + 2 * (self.bbox_padding_y - self.bbox_padding_x))
            self.ax_ylabel_boxes[visualized_info_df_entry['Task']].set_boxstyle("Round", pad=self.bbox_padding_x)

            self.ax_ylabel_labels[visualized_info_df_entry['Task']].set_x(self.padding_left - self.bbox_width/2 - self.bbox_padding_x)
            self.ax_ylabel_labels[visualized_info_df_entry['Task']].set_y(bbox_y + (bbox_height-2*self.bbox_padding_y)/2)


        bbox_height = height_ratio[0] / sum(height_ratio) * (self.padding_top-self.padding_bottom)
        bbox_node_indexes = [key for key, box in self.ax_ylabel_boxes.items() if (box.get_visible() and isinstance(key, int)) ]
        tasks_per_Node = [len(settings.info_dataframe) for i in range(len(bbox_node_indexes)-1)] + [len(settings.info_dataframe)-1]
        bbox_height_Nodes = [x / sum(tasks_per_Node) * bbox_height for x in tasks_per_Node]
        bbox_y = self.padding_top + self.bbox_padding_x

        for i in bbox_node_indexes:
            bbox_y -= bbox_height_Nodes[i]
            self.ax_ylabel_boxes[i].set_x(self.padding_left - self.bbox_width - self.bbox_padding_x)
            self.ax_ylabel_boxes[i].set_y(bbox_y)
            self.ax_ylabel_boxes[i].set_width(self.bbox_width)
            self.ax_ylabel_boxes[i].set_height(bbox_height_Nodes[i]-2*self.bbox_padding_y + 2 * (self.bbox_padding_y - self.bbox_padding_x))
            self.ax_ylabel_boxes[i].set_boxstyle("Round", pad=self.bbox_padding_x)

            self.ax_ylabel_labels[i].set_x(self.padding_left - self.bbox_width/2 - self.bbox_padding_x)
            self.ax_ylabel_labels[i].set_y(bbox_y + (bbox_height_Nodes[i]-2*self.bbox_padding_y)/2)

        if hasattr(self, 'simulation_elem'):
            self.name_Bars()
        plt.draw()

    def update_ax_ylabel_labels(self):
        for info_df_index, info_df_entry in settings.info_dataframe.iterrows():
            text = "Mean " + info_df_index + " load\n(rel. to " + str(info_df_entry['Number of Channel'] * info_df_entry['Max. Data Rate']) + ")"
            text = textwrap.fill(text, width=15)
            self.ax_ylabel_labels[info_df_entry['Task']].set_text(text)

    def reset_name_bars(self):
        for index, name_label in self.ax_text_tasks.items():
            if name_label is not None:
                name_label.set_visible(False)
        self.ax_text_tasks = {}

        for index, name_label in self.ax_text_latencies.items():
            if name_label is not None:
                name_label.set_visible(False)
        self.ax_text_latencies = {}

    def color(self, row):
        return settings.color_dict[row['Hardwareclass']]

    ##
    # function creates a simulation element and draws the results in the graph
    def generate_results(self):
        if hasattr(self, 'simulation_elem'):
            self.reset_figure()
        
        self.simulation_elem = simulation.Simulation()
        
        self.simulation_elem.timing_dataframe['color'] = self.simulation_elem.timing_dataframe.apply(self.color, axis=1)

        for i in range(self.simulation_elem.dimension+1):
            self.ax_ylabel_boxes[i].set_visible(True)
            self.ax_ylabel_labels[i].set_visible(True)

        self.plot_data_rate()
        
        self.rects = self.ax_figure.barh(self.simulation_elem.timing_dataframe.index[::-1],
                                         self.simulation_elem.timing_dataframe.Duration,
                                         left=self.simulation_elem.timing_dataframe.Start,
                                         color=self.simulation_elem.timing_dataframe.color,
                                         edgecolor=settings.hightlight_edge_color,
                                         linewidth=0,
                                         height=0.95)
        self.ax_figure.set_ylim([self.rects[-1].get_y(), self.rects[0].get_y()+self.rects[0].get_height()])
        
        self.ax_figure_lines = list((None,)*len(self.rects))
        self.ax_data_rate_lines = list((None,)*len(self.rects))
        # Marking the current date on the chart
        data_rates_to_be_plotted = settings.info_dataframe.loc[settings.info_dataframe.visualize==True, 'Task'].values.tolist()
        for index, dataframe_entry in self.simulation_elem.timing_dataframe.loc[self.simulation_elem.timing_dataframe['Task'].isin(data_rates_to_be_plotted)][['Task', 'Start', 'End', 'color']].iterrows():
            self.ax_figure_lines[index] = ( self.ax_figure.axvline(dataframe_entry['Start'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3), self.ax_figure.axvline(dataframe_entry['End'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3) )
            self.ax_data_rate_lines[index] = ( self.ax_data_rate[dataframe_entry['Task']].axvline(dataframe_entry['Start'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3), self.ax_data_rate[dataframe_entry['Task']].axvline(dataframe_entry['End'], color=dataframe_entry['color'], linestyle='dashed', linewidth=0.3) )

        self.simulation_elem.latency_dataframe['color'] = self.simulation_elem.latency_dataframe.apply(self.color, axis=1)
        self.latency_rects = self.ax_figure.barh(self.simulation_elem.latency_dataframe['Position'], self.simulation_elem.latency_dataframe.Duration, left=self.simulation_elem.latency_dataframe.Start, color=self.simulation_elem.latency_dataframe.color, edgecolor=settings.hightlight_edge_color, linewidth=0)
        self.plot_duration_extension()
        self.update_axes()
        self.update_ax_ylabel_labels()
        
        ##### LEGENDS #####
        legend_elements = [patches.Patch(facecolor=settings.color_dict[i], label=i)  for i in settings.color_dict]
        self.ax_figure.legend(handles=legend_elements, loc="upper right", shadow=False, fontsize='x-small', frameon=True) 

        plt.draw()

    def plot_data_rate(self):
        temp_dataframe = settings.info_dataframe
        temp_dataframe['color'] = temp_dataframe.apply(self.color, axis=1)
        
        self.data_rate_zeroline = {}
        for task, used_color, current_max_rel_usable_data_rate in temp_dataframe.loc[:, ['Task', 'color', 'max. rel. usable Data_Rate']].values.tolist():
            x_values = [0, self.simulation_elem.total_latency]
            self.data_rate_zeroline[task] = self.ax_data_rate[task].plot(x_values, [0,0], linewidth=0.7, color=used_color)

            y_values = [current_max_rel_usable_data_rate, current_max_rel_usable_data_rate]
            self.ax_data_rate[task].plot(x_values, y_values, linestyle="--", linewidth=0.7, color=settings.max_marker_color)
            text = "max"
            if current_max_rel_usable_data_rate < 0.7:
                self.ax_data_rate[task].text(x_values[1], current_max_rel_usable_data_rate, text, ha='right', va='bottom', color=settings.max_marker_color, fontsize='xx-small')
            else:
                self.ax_data_rate[task].text(x_values[1], current_max_rel_usable_data_rate, text, ha='right', va='top', color=settings.max_marker_color, fontsize='xx-small')

            for index in range(len(self.simulation_elem.ideal_rel_data_rate[task])):
                duration = self.simulation_elem.ideal_uncombined_intervals[task][index][1] - self.simulation_elem.ideal_uncombined_intervals[task][index][0]
                height = self.simulation_elem.ideal_rel_data_rate[task][index]
                rect = patches.Rectangle((self.simulation_elem.ideal_uncombined_intervals[task][index][0], 0), duration, height, facecolor=settings.duration_extension_color)
                self.ax_data_rate[task].add_patch(rect)
                
            if len(self.simulation_elem.ideal_rel_data_rate[task]) > 0:
                index_min = np.argmax(self.simulation_elem.ideal_rel_data_rate[task])
                x_values = [self.simulation_elem.ideal_uncombined_intervals[task][index_min][0], self.simulation_elem.ideal_uncombined_intervals[task][index_min][1]]
                self.ax_data_rate[task].text(np.mean(x_values), current_max_rel_usable_data_rate, str(int( round(max(self.simulation_elem.ideal_rel_data_rate[task]), 2) *100)) + "%", ha='center', va='bottom', color='white', fontsize='xx-small')

            for index in range(len(self.simulation_elem.final_intervals[task])):

                duration = self.simulation_elem.final_intervals[task][index][1] - self.simulation_elem.final_intervals[task][index][0]
                height = self.simulation_elem.final_rel_data_rate[task][index]
                rect = patches.Rectangle((self.simulation_elem.final_intervals[task][index][0], 0), duration, height, facecolor=used_color)
                self.ax_data_rate[task].add_patch(rect)

                x_values = [self.simulation_elem.final_intervals[task][index][0], self.simulation_elem.final_intervals[task][index][1]]
                y_value = self.simulation_elem.final_rel_data_rate[task][index]

                if y_value >= min(current_max_rel_usable_data_rate, 0.5):
                    self.ax_data_rate[task].text(np.mean(x_values), (self.simulation_elem.final_rel_data_rate[task][index]-0.15), str(int( round(y_value, 2) *100)) + "%", ha='center', va='center', color='white', fontsize='xx-small')
                else:
                    if y_value < 0.1:
                        text = str(round(y_value*1e2, 1) ) + "%"
                    else:
                        text = str(int(y_value*1e2) ) + "%"
                    self.ax_data_rate[task].text(np.mean(x_values), (self.simulation_elem.final_rel_data_rate[task][index]+0.15), text, ha='center', va='center', color='white', fontsize='xx-small')

    def plot_duration_extension(self):
        self.extension_rect = {}

        for index, data_type in self.simulation_elem.extension_dict.items():
                self.extension_rect[index] = patches.Rectangle((self.rects[index].get_x()+data_type[0], self.rects[index].get_y()), data_type[1], self.rects[index].get_height(), facecolor=settings.duration_extension_color, edgecolor=settings.hightlight_edge_color, linewidth=0)
                self.ax_figure.add_patch(self.extension_rect[index])

    ##
    # function labels the bars in the graph
    def name_Bars(self):
        for index, data_type in self.simulation_elem.timing_dataframe.iterrows():
            if len(self.ax_text_tasks) > 0 and index in self.ax_text_tasks and self.ax_text_tasks[index].get_visible():
                self.ax_text_tasks[index].remove()
            text = data_type['Task']
            fontsizes = [18.0, 16.0, 14.0, 12.0, 10.0, 8.0, 6.0, 5.0, 4.0]
            self.ax_text_tasks[index] = self.test_fontsizes(fontsizes, index, text, self.rects, True)
            if self.ax_text_tasks[index] is None:
                del self.ax_text_tasks[index]

        for index, data_type in self.simulation_elem.latency_dataframe.iterrows():
            if len(self.ax_text_latencies) > 0 and index in self.ax_text_latencies and self.ax_text_latencies[index].get_visible():
                self.ax_text_latencies[index].remove()
            text = data_type['Hardwareclass']
            fontsizes = [10.0, 8.0, 6.0, 5.0, 4.0]
            self.ax_text_latencies[index] = self.test_fontsizes(fontsizes, index, text, self.latency_rects, False)
            if self.ax_text_latencies[index] is None:
                del self.ax_text_latencies[index]
        plt.draw()

    ##
    # function tests if the label fits into the rects and returns the text element which fits
    def test_fontsizes(self, fontsizes, index, text, rect_list, test_line_break):
        for font_size in fontsizes:
            ax_text = self.ax_figure.text(rect_list[index].get_center()[0], rect_list[index].get_center()[1], text, ha='center', va='center', color='black', fontsize=font_size)

            text_width = max(ax_text.get_window_extent().intervalx) - min(ax_text.get_window_extent().intervalx)
            text_height = max(ax_text.get_window_extent().intervaly) - min(ax_text.get_window_extent().intervaly)
            rect_width = max(rect_list[index].get_window_extent().intervalx) - min(rect_list[index].get_window_extent().intervalx)
            rect_height = max(rect_list[index].get_window_extent().intervaly) - min(rect_list[index].get_window_extent().intervaly)

            if text_width+self.name_bar_padding > rect_width or text_height+self.name_bar_padding > rect_height:
                ax_text.remove()
                if test_line_break and text_height+self.name_bar_padding < rect_height:
                    ax_text = self.test_line_break_name_bar(font_size, index, text, rect_width, rect_height)
                    if ax_text is not None:
                        return ax_text
            else:
                return ax_text
        return None

    ##
    # function tests if the label with line break fits into the rects and returns the text element that fits
    def test_line_break_name_bar(self, fontsize, index, text, rect_width, rect_height):
        pos = text.find(" ", 20)
        if pos != -1:
            text = text[:pos] + "\n" + text[pos+1:]
        ax_text = self.ax_figure.text(self.rects[index].get_center()[0], self.rects[index].get_center()[1], text, ha='center', va='center', color='black', fontsize=fontsize)
        text_width = max(ax_text.get_window_extent().intervalx) - min(ax_text.get_window_extent().intervalx)
        text_height = max(ax_text.get_window_extent().intervaly) - min(ax_text.get_window_extent().intervaly)
        if text_width+self.name_bar_padding > rect_width or text_height+self.name_bar_padding > rect_height:
            ax_text.remove()
            return None
        else:
            return ax_text

    def save_plot(self, file_path):
        file_extension = os.path.splitext(file_path)[1]

        if file_path:
            if file_extension == ".pdf":
                self.fig.savefig(file_path, format='pdf', dpi=900)
                print("Figure was saved as a PDF.")
            elif file_extension == ".svg":
                self.fig.savefig(file_path, format='svg', dpi=900)
                print("Figure was saved as an SVG.")
            elif file_extension == ".txt":
                settings.save_settings(file_path)
                print("Settings were saved as a TXT.")
            elif file_extension == ".png":
                self.fig.savefig(file_path, format='png', dpi=900)
                print("Figure was saved as a PNG.")
            elif file_extension == ".jpg" or file_extension == ".jpeg":
                self.fig.savefig(file_path, format='jpeg', dpi=900)
                print("Figure was saved as a JPG.")
            else:
                print("Invalid data type.")
