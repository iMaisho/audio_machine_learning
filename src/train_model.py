import os
os.environ["KERAS_BACKEND"] = "torch"
import keras

def train_model(genres, features_train, labels_train, features_val, labels_val, epoch = 64):

    _, b = features_train.shape

    inputs = keras.Input(shape=(b,), name="feature")
    x = keras.layers.Dense(300, activation="relu", name="dense_1")(inputs)
    x = keras.layers.Dense(200, activation="relu", name="dense_2")(x)
    outputs = keras.layers.Dense(len(genres), activation="softmax", name="predictions")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        # Optimizer
        optimizer=keras.optimizers.RMSprop(),
        # Loss function to minimize
        loss=keras.losses.SparseCategoricalCrossentropy(),
        # List of metrics to monitor
        metrics=[keras.metrics.SparseCategoricalAccuracy()],
    )
    model.fit(x=features_train, y=labels_train, verbose=1, validation_data=(features_val, labels_val), epochs=epoch)