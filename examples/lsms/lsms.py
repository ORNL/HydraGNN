import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "lsms.json")
hydragnn.run_training(filepath)
