from train import train

config = dict(
    optimizer='Adam',
    optimizer_learning_rate=0.0001,
    batch_size=1000,
    num_epochs=250,
    seed=42,
    model='simple_conv'
)

train(config)

