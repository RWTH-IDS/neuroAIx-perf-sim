
import gui_window
import settings
import argparse
import simulation
import result_figure
import os
import sys

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", nargs="?", help="Direction of settings txt file", default="microcircuit.txt")
    parser.add_argument("--no_gui", action="store_true", help="Deactivates the GUI")
    parser.add_argument('--save_fig', nargs='*', default=False, help='Saves the figure if no_gui is set (optional name) (optional file extension)')
    parser.add_argument("--print_info", choices=["none", "standard", "detailed"], default=False, help="Specifies the level of information to print")
    args = parser.parse_args()

    # args.print_info has a different default parameter depending on args.no_gui
    if args.print_info is False:
        args.print_info = "none"
        if args.no_gui:
            args.print_info = "standard"

    settings.initialize_variables(args.file_path, args.no_gui, args.print_info)

    try:
        if args.save_fig is not False:
            if len(args.save_fig) == 0:
                file_name_without_extension = os.path.splitext(args.file_path)[0]
                fig_filename = file_name_without_extension + '.' + settings.default_figure_file_extension

            elif len(args.save_fig) == 1:
                if args.save_fig[0] in settings.supported_figure_file_extensions:
                    file_name_without_extension = os.path.splitext(args.file_path)[0]
                    fig_filename = file_name_without_extension + '.' + args.save_fig[0]
                else:
                    fig_filename = args.save_fig[0] + '.' + settings.default_figure_file_extension

            elif len(args.save_fig) == 2:
                    if args.save_fig[0] in settings.supported_figure_file_extensions:
                        fig_filename = args.save_fig[1] + '.' + args.save_fig[0]
                    elif args.save_fig[1] in settings.supported_figure_file_extensions:
                        fig_filename = args.save_fig[0] + '.' + args.save_fig[1]
                    else:
                        raise Exception("Unsupported file extension")

            else:
                raise Exception("Error processing the argument --save_fig: Too many arguments")

    except Exception as e:
        print(f"Error processing the argument --save_fig: {str(e)}")
        sys.exit(1)

    if args.no_gui:
        settings.update_variables()
        if args.save_fig is not False:
            result_figure = result_figure.Result_figure()
            result_figure.generate_results()
            result_figure.save_plot(fig_filename)
        else:
            simulation_elem = simulation.Simulation()
    else:
        gui_window.App('Network Simulator', (1200,840))
