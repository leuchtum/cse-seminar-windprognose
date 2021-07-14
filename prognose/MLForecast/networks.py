import tensorflow as tf
import wandb


def decode_layer(layer_type, config):
    layer_type = layer_type.split("_")

    # DECODE CONFIG
    if layer_type[0] in config:
        for key, instruction in config[layer_type[0]].items():
            if type(instruction) is not str:
                continue
            instruction = instruction.split("_")

            # L1 Reg
            if instruction[0] == "L1":
                l1 = float(instruction[1])
                if l1 > 0:
                    val = tf.keras.regularizers.L1(l1=l1)
                
            # L2 Reg
            if instruction[0] == "L2":
                l2 = float(instruction[1])
                if l2 > 0:
                    val = tf.keras.regularizers.L2(l2=l2)

            # Update config
            config[layer_type[0]][key] = val

    # DECODE LAYER
    cfg = {}

    # Dropout
    if layer_type[0] == "DROP":
        val = config[layer_type[1]]
        return tf.keras.layers.Dropout(val)

    # LSTM
    if layer_type[0] == "LSTM":
        val = config[layer_type[1]]
        if "LSTM" in config:
            cfg = config["LSTM"]
        return tf.keras.layers.LSTM(val, **cfg)

    # Dense
    if layer_type[0] == "DENSE":
        val = config[layer_type[1]]
        if "DENSE" in config:
            cfg = config["DENSE"]
        return tf.keras.layers.Dense(val, **cfg)


class ModelBase:
    def __init__(self):
        self.name = None

    def set_input_shape(self, shape):
        self.input_shape = shape

    def set_output_shape(self, shape):
        self.output_shape = shape


class ModelNN(ModelBase):
    def __init__(self):
        super().__init__()
        self.optimizer = tf.optimizers.Adam()
        self.model = tf.keras.models.Sequential()

        self.add_loss_and_metrics("mse")  # Standard

    def add_loss_and_metrics(self, metric, y_width=None, shift=None, factor=0):
        if metric == "mae":
            self.loss = tf.metrics.MeanAbsoluteError()
        elif metric == "mse":
            self.loss = tf.metrics.MeanSquaredError()
        elif metric == "rmse":
            self.loss = tf.metrics.RootMeanSquaredError()

        self.loss_weights = None
        if bool(shift) ^ bool(y_width):  # XOR
            raise(Exception("Specify shift and y_width"))
        if shift and y_width:
            self.loss_weights = [1 for _ in range(y_width)]
            self.loss_weights[:-shift] = [i *
                                          factor for i in self.loss_weights[:-shift]]

        self.metrics = [tf.metrics.RootMeanSquaredError(),
                        tf.metrics.MeanAbsoluteError()]

    def fit(self, train, val, epochs=10, batch_size=32, early_stop=None, save=False, track_wandb=False):
        callbacks = []

        if early_stop:
            callbacks.append(tf.keras.callbacks.EarlyStopping(
                patience=early_stop,
                monitor='val_loss',
                mode='min'
            ))

        if track_wandb:
            # check if wandb is init
            if hasattr(wandb.config, "as_dict"):
                callbacks.append(wandb.keras.WandbCallback())

        if save:
            pass

        self.history = self.model.fit(
            x=train[0],
            y=train[1],
            validation_data=val,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )

    def compile(self):
        self.model.compile(
            loss=self.loss,
            optimizer=self.optimizer,
            metrics=self.metrics,
            #loss_weights=self.loss_weights
        )


class LSTMseq2seq(ModelNN):
    def __init__(self, structure, config):
        super().__init__()
        self.structure = structure
        self.config = config

    def build(self):
        assert self.input_shape[0] == self.output_shape[0]

        self.model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))

        for layer in self.structure:
            if layer != "OUT":
                cfg = self.config
                if "LSTM" not in cfg:
                    cfg["LSTM"] = {}
                cfg["LSTM"].update({"return_sequences": True})
                self.model.add(decode_layer(layer, cfg))
            else:
                self.model.add(tf.keras.layers.Dense(
                    self.output_shape[1], name="out_dense"))


class LSTMseq2vec(ModelNN):
    def __init__(self, structure, config):
        super().__init__()
        self.structure = structure
        self.config = config

    def build(self):
        self.model.add(tf.keras.layers.InputLayer(input_shape=self.input_shape))

        for i in range(len(self.structure)):
            if "LSTM" in self.structure[i]:
                last_lstm_index = i

        for i in range(len(self.structure)):
            if self.structure[i] != "OUT":
                cfg = self.config
                if "LSTM" not in cfg:
                    cfg["LSTM"] = {}
                if i < last_lstm_index:
                    cfg["LSTM"].update({"return_sequences": True})
                else:
                    cfg["LSTM"].update({"return_sequences": False})
                self.model.add(decode_layer(self.structure[i], cfg))
            else:
                n_neurons = self.output_shape[0] * self.output_shape[1]
                self.model.add(tf.keras.layers.Dense(
                    n_neurons, name="out_dense"))
                self.model.add(tf.keras.layers.Reshape(self.output_shape))


if __name__ == "__main__":
    structure = ["LSTM_x", "DROP_d", "LSTM_y", "DROP_d", "DENSE_y", "OUT"]
    config = {
        "LSTM": {
            "recurrent_regularizer": "L1_0.0001"
        },
        "x": 32,
        "y": 10,
        "d": 0.2
    }

    my_model = LSTMseq2vec(structure, config)
    my_model.set_input_shape((24, 10))
    my_model.set_output_shape((24, 2))
    my_model.add_loss_and_metrics("mae", y_width=24, shift=12)
    my_model.build()
    my_model.compile()

    train = (1, 1)
    val = (2, 2)
