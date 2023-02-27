# Only ask users to install matplotlib if they actually need it
from __future__ import annotations

try:
    import matplotlib.pyplot as plt
except ImportError:
    raise ImportError(
        "To display the environment in a window, please install matplotlib, eg: `pip3 install --user matplotlib`"
    )
try:
    import pygame
except ImportError:
    raise ImportError(
        "To display the environment using pygame, please install pygame, eg: `pip3 install --user pygame`"
    )


class Window:
    """
    Window to draw a gridworld instance using Matplotlib
    """

    def __init__(self, title, display_mode="matplotlib"):
        self.no_image_shown = True

        # Create the figure and axes
        self.fig, self.ax = plt.subplots()

        # Show the env name in the window title
        self.fig.canvas.manager.set_window_title(title)

        # Turn off x/y axis numbering/ticks
        self.ax.xaxis.set_ticks_position("none")
        self.ax.yaxis.set_ticks_position("none")
        _ = self.ax.set_xticklabels([])
        _ = self.ax.set_yticklabels([])

        # Flag indicating the window was closed
        self.closed = False

        def close_handler(evt):
            self.closed = True

        self.fig.canvas.mpl_connect("close_event", close_handler)

        self.display_mode = display_mode
        if self.display_mode == "pygame":
            self.pygame_window = pygame.display.set_mode((640, 480))
            self.pygame_screen = pygame.display.get_surface()
            pygame.init()
            pygame.display.set_caption("Minigrid")

    def show_img(self, img):
        """
        Show an image or update the image being shown
        """

        # If no image has been shown yet,
        # show the first image of the environment
        if self.no_image_shown:
            self.imshow_obj = self.ax.imshow(img, interpolation="bilinear")
            self.no_image_shown = False
        # Update the image data
        self.imshow_obj.set_data(img)

        if self.display_mode == "pygame":
            self.fig.canvas.draw()
            renderer = self.fig.canvas.get_renderer()
            raw_data = renderer.tostring_rgb()

            size = self.fig.canvas.get_width_height()
            surf = pygame.image.fromstring(raw_data, size, "RGB")

            self.pygame_screen.blit(surf, (0, 0))
            pygame.display.flip()

        else:
            # Request the window be redrawn
            self.fig.canvas.draw_idle()
            self.fig.canvas.flush_events()

            # Let matplotlib process UI events
            plt.pause(0.001)

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
        self.fig.canvas.mpl_connect("key_press_event", key_handler)
        self.key_handler = key_handler

    def show(self, block=True):
        """
        Show the window, and start an event loop
        """

        if self.display_mode == "pygame":
            while not self.closed:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        self.closed = True
                    if event.type == pygame.KEYDOWN:
                        event.key = pygame.key.name(event.key)
                        self.key_handler(event)

        else:
            # If not blocking, trigger interactive mode
            if not block:
                plt.ion()

            # Show the plot
            # In non-interative mode, this enters the matplotlib event loop
            # In interactive mode, this call does not block
            plt.show()

    def close(self):
        """
        Close the window
        """
        if self.display_mode == "pygame":
            pygame.quit()
        else:
            plt.close()
        self.closed = True
