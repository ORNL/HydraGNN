import torch
from ray import tune

from data_utils.dataset_descriptors import (
    AtomFeatures,
    StructureFeatures,
)


def train_validate_test_hyperopt(
    config, checkpoint_dir=None, data_dir=None, writer=None
):
    atom_features = [
        AtomFeatures.NUM_OF_PROTONS,
        AtomFeatures.CHARGE_DENSITY,
    ]
    structure_features = [StructureFeatures.FREE_ENERGY]

    input_dim = len(atom_features)
    perc_train = 0.7
    dataset1, dataset2 = load_data(config, structure_features, atom_features)

    model = generate_model(
        model_type="PNN",
        input_dim=input_dim,
        dataset=dataset1[: int(len(dataset1) * perc_train)],
        config=config,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["learning_rate"])
    scheduler = ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10, min_lr=0.00001
    )

    train_loader, val_loader, test_loader = combine_and_split_datasets(
        dataset1=dataset1,
        dataset2=dataset2,
        batch_size=config["batch_size"],
        perc_train=perc_train,
    )

    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda:0"
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
    model.to(device)

    if checkpoint_dir:
        model_state, optimizer_state = torch.load(
            os.path.join(checkpoint_dir, "checkpoint")
        )
        model.load_state_dict(model_state)
        optimizer.load_state_dict(optimizer_state)

    num_epoch = 200

    for epoch in range(0, num_epoch):
        train_mae = train(train_loader, model, optimizer)
        val_mae = validate(val_loader, model)
        test_rmse = test(test_loader, model)
        scheduler.step(val_mae)
        print(
            f"Epoch: {epoch:02d}, Train MAE: {train_mae:.4f}, Val MAE: {val_mae:.4f}, "
            f"Test RMSE: {test_rmse:.4f}"
        )
        if epoch % 10 == 0:
            with tune.checkpoint_dir(epoch) as checkpoint_dir:
                path = os.path.join(checkpoint_dir, "checkpoint")
                torch.save((model.state_dict(), optimizer.state_dict()), path)

        tune.report(train_mae=train_mae, val_mae=val_mae, test_rmse=test_rmse)
