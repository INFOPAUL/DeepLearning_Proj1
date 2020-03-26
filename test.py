import train
import train_model_no_WS

config_simple = dict(
    optimizer='Adam',
    optimizer_learning_rate=0.001,
    batch_size=128,
    num_epochs=250,
    model='simple_conv',
    class_num=2,
    channels_in=2,
    augmentation=False
)

#train.train(config_simple)


config_model_no_WS = dict(
    optimizer='Adam',
    optimizer_learning_rate=0.001,
    batch_size=128,
    num_epochs=250,
    model='simple_conv',
    class_num=10,
    channels_in=1,
    augmentation=False
)
#train_model_no_WS.train(config_model_no_WS)

config_siamese = dict(
    optimizer='Adam',
    optimizer_learning_rate=0.001,
    batch_size=128,
    num_epochs=250,
    model='siamese',
    class_num=2,
    channels_in=1,
    augmentation=False
)

#train.train(config_siamese)

config_siamese_no_WS = dict(
    optimizer='Adam',
    optimizer_learning_rate=0.001,
    batch_size=128,
    num_epochs=250,
    model='siamese_no_WS',
    class_num=2,
    channels_in=1,
    augmentation=False
)

train.train(config_siamese_no_WS)