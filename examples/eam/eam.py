import os
import hydragnn

filepath = os.path.join(os.path.dirname(__file__), "NiNb_EAM_bulk_multitask.json")
hydragnn.run_training(filepath)
