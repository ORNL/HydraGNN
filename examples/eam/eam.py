import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "eam_energy.json")
hydragnn.run_training(filepath)
