import matplotlib.pyplot as plt
import os

class Plotter:
    """
    A matplotlib plotting class
    """

    def __init__(self, save_dir='results/figures'):
        """
        Initialize plotter.

        Parameters
        ----------
        save_dir : str
            Directory to save figures.
        """

        self.save_dir = save_dir
        # Create folder if it does not exist
        os.makedirs(self.save_dir, exist_ok=True)

    