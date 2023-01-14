class LSTMConfig:
    embeds_size = 128
    hidden_size = 128
    n_layers = 1
    crf = True


class TrainingConfig:
    epochs = 30
    batch_size = 1024
    lr = 0.001
