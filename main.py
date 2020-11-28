import torch
from ray import tune
from ray.tune import CLIReporter
from ray.tune.suggest.bohb import TuneBOHB
from ray.tune.schedulers import HyperBandForBOHB
import numpy as np
from utilities.utils import train_validate_test
from functools import partial
import os

os.environ["SERIALIZED_DATA_PATH"] = os.getcwd()

config = {"batch_size": tune.choice([8,16,32,64]),
         "learning_rate": tune.choice([0.01, 0.005, 0.001, 0.0005, 0.0001]),
         "num_conv_layers": tune.randint(5,20),
         "hidden_dim": tune.choice([15,20,25,30,35]),
         "radius": tune.randint(2,25),
         "max_num_node_neighbours": tune.randint(1, 32),
         }

algo = TuneBOHB(max_concurrent=10, metric="val_mae", mode="min")

bohb = HyperBandForBOHB(
    time_attr="training_iteration",
    metric="val_mae",
    mode="min",
    max_t=200)

reporter = CLIReporter(
    metric_columns=["train_mae", "val_mae", "training_iteration"])

result = tune.run(
    partial(train_validate_test, checkpoint_dir="./checkpoint-ray-tune"),
    resources_per_trial={"gpu": 0.1},
    config=config,
    search_alg=algo,
    num_samples=100,
    scheduler=bohb,
    progress_reporter=reporter)

'''
model setup
model_choices = {"1": "GIN", "2": "PNN", "3": "GAT", "4": "MFC"}
print("Select which model you want to use: 1) GIN 2) PNN 3) GAT 4) MFC")
chosen_model = model_choices[input("Selected value: ")]
model = generate_model(model_type=chosen_model, input_dim=input_dim, dataset=dataset, max_num_node_neighbours=max_num_node_neighbours)

model_name = (
    model.__str__()
    + "-r-"
    + str(radius)
    + "-mnnn-"
    + str(max_num_node_neighbours)
    + "-num_conv_layers-"
    + str(model.num_conv_layers)
    + "-hd-"
    + str(model.hidden_dim)
    + "-ne-"
    + str(num_epoch)
    + "-lr-"
    + str(learning_rate)
    + "-bs-"
    + str(batch_size)
    + ".pk"
)
writer = SummaryWriter("./logs/" + model_name)
train_validate_test(
    model,
    optimizer,
    num_epoch,
    train_loader,
    val_loader,
    test_loader,
    writer,
    scheduler,
)
torch.save(model.state_dict(), "./models_serialized/" + model_name)

model.load_state_dict(torch.load("models_serialized/PNNStack7-mnnn-5-hd-75-ne-200-lr-0.01.pk", map_location=torch.device('cpu')))
print(test(test_loader, model))


'''
