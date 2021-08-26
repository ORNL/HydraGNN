import matplotlib.pyplot as plt
from itertools import chain
import time
import numpy as np


class Visualizer:
    """A class used for visualizing values in a scatter plot. There are two attributes: true_values and predicted_values that we want to see
    in a scatter plot. The ideal case is that the values will be positioned on a thin diagonal of the scatter plot.

    Methods
    -------
    add_test_values(true_values: [], predicted_values: [])
        Add the true and predicted values to the lists.
    create_scatter_plot()
        Create the scatter plot out of true and predicted values.
    """

    def __init__(self, model_with_config_name: str):
        self.true_values = []
        self.predicted_values = []
        self.model_with_config_name = model_with_config_name

    def add_test_values(self, true_values: [], predicted_values: []):
        """Append true and predicted values to existing lists.

        Parameters
        ----------
        true_values: []
            List of true values to append to existing one.
        predicted_values: []
            List of predicted values to append to existing one.
        """
        self.true_values.extend(true_values)
        self.predicted_values.extend(predicted_values)

    def __convert_to_list(self):
        """When called it performs flattening of a list because the values that are stored in true and predicted values lists are in
        the shape: [[1], [2], ...] and in order to visualize them in scatter plot they need to be in the shape: [1, 2, ...].
        """
        if len(self.true_values) * len(self.true_values[0]) != len(
            self.predicted_values
        ) * len(self.predicted_values[0]):
            print("Length of true and predicted values array is not the same!")

        self.true_values = list(chain.from_iterable(self.true_values))
        self.predicted_values = list(chain.from_iterable(self.predicted_values))

    def create_scatter_plot(self, save_plot=True):
        """Creates scatter plot from stored values in the tru and  predicted values lists."""
        self.__convert_to_list()
        plt.scatter(self.true_values, self.predicted_values)
        plt.title("Scatter plot for true and predicted values.")
        plt.xlabel("True value")
        plt.ylabel("Predicted value")
        if save_plot:
            plt.savefig(
                f"./logs/{self.model_with_config_name}/scatter_plot_test_{self.model_with_config_name}.png"
            )
            plt.close()
        else:
            plt.show()
