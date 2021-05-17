from functools import partial

from ray import tune
from hyperopt import hp
from ray.tune import CLIReporter
from ray.tune.suggest import ConcurrencyLimiter
from ray.tune.suggest.hyperopt import HyperOptSearch
from ray.tune.schedulers import ASHAScheduler

from utils.tune import train_validate_test_hyperopt


def run_with_hyperparameter_optimization():
    config = {
        "batch_size": hp.choice("batch_size", [64]),
        "learning_rate": hp.choice("learning_rate", [0.005]),
        "num_conv_layers": hp.choice("num_conv_layers", [8, 10, 12, 14]),
        "hidden_dim": hp.choice("hidden_dim", [20]),
        "radius": hp.choice("radius", [5, 10, 15, 20, 25]),
        "max_num_node_neighbours": hp.choice(
            "max_num_node_neighbours", [5, 10, 15, 20, 25, 30]
        ),
    }

    algo = HyperOptSearch(space=config, metric="test_mae", mode="min")
    algo = ConcurrencyLimiter(searcher=algo, max_concurrent=10)

    scheduler = ASHAScheduler(
        time_attr="training_iteration",
        metric="test_mae",
        mode="min",
        grace_period=10,
        reduction_factor=3,
    )

    reporter = CLIReporter(
        metric_columns=["train_mae", "val_mae", "test_mae", "training_iteration"]
    )

    result = tune.run(
        partial(train_validate_test_hyperopt, checkpoint_dir="./checkpoint-ray-tune"),
        resources_per_trial={"cpu": 0.5, "gpu": 0.1},
        search_alg=algo,
        num_samples=100,
        scheduler=scheduler,
        progress_reporter=reporter,
    )


if __name__ == "__main__":
    run_with_hyperparameter_optimization()
