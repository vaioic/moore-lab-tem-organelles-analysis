from shared.organelle_detector import train

train(
    "../data/Dataset01/clean/mouse192.csv",
    "../test_train",
    model_name="test_1",
    validation_data_path=None,
    num_epochs=5,
)
