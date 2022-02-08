import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "fept_multi.json")
hydragnn.run_training(filepath)
