from pydantic import BaseModel


# parse example:
# Optimizer.model_validate_json( '{ "learn_rate": 0.0001, "batch_size": 128 }' )
class OptimizerModel(BaseModel):
    learning_rate: float
    batch_size: int


class TrainingModel(BaseModel):
    Optimizer: OptimizerModel


class NeuralNetworkModel(BaseModel):
    layers: int


class Config(BaseModel):
    NeuralNetwork: NeuralNetworkModel
    Training: TrainingModel


"""
Note: Validation errors look like

NeuralNetwork.Training.Optimizer.learning_rate
  Field required [type=missing, input_value={'learn_rate': 0.001, 'batch_size': 128}, input_type=dict]

which is saying that it expected to find "learning_rate",
but saw instead, "{'learn_rate': 0.001, 'batch_size': 128}"
"""


def main(argv):
    assert len(argv) == 2, f"{argv[0]} <config.json>"
    with open(argv[1], encoding="utf-8") as f:
        cfg = Config.model_validate_json(f.read())
    print(cfg)


if __name__ == "__main__":
    import sys

    main(sys.argv)
