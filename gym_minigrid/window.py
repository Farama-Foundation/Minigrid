import sys
import numpy as np

# Only ask users to install matplotlib if they actually need it
try:
    import matplotlib.pyplot as plt
except:
    print('To display the environment in a window, please install matplotlib, eg:')
    print('pip3 install --user matplotlib')
    sys.exit(-1)

class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title):
        self.fig = None

        self.imshow_obj = None

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.set_xticks([], [])
        self.ax.set_yticks([], [])

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # Show the first image of the environment
        if self.imshow_obj is None:
            self.imshow_obj = self.ax.imshow(img, interpolation='bilinear')

        self.imshow_obj.set_data(img)
        self.fig.canvas.draw()

    def set_caption(self, text):
        """
        Set/update the caption text below the image
        """

        plt.xlabel(text)

    def reg_key_handler(self, key_handler):
        """
        Register a keyboard event handler
        """

        # Keyboard handler
        self.fig.canvas.mpl_connect('key_press_event', key_handler)

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        # If not blocking, trigger interactive mode
        if not block:
            plt.ion()

        # Show the plot, enter the matplotlib event loop
        plt.show(block=block)

    def close(self):
        """
        Close the window
        """

        plt.close()
