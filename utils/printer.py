import datetime
from termcolor import colored


class Printer:
    def __init__(self, color="white"):
        self.module_color_map = {}
        # Define a list of colors available in termcolor
        colors = ["white", "red", "green", "yellow", "blue", "magenta", "cyan"]
        # Assign a color to each module name, cycling through the list of colors
        for i, module_name in enumerate(module_names):
            self.module_color_map[module_name] = colors[i % len(colors)]

    def print(self, module_name="printer", message: str=""):
        # Get current timestamp
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Get the color assigned to the module name
        color = self.module_color_map.get(module_name, "white")
        # Format the output string
        output = f"[{module_name}][{timestamp}] {message}"
        # Print the colored output
        print(colored(output, color))
