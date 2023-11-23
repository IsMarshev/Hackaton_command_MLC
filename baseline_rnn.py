import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa 
import os

PATH = os.getcwd()
LSTM_SIZE = 256
TIMESTEP = 128
NUM_CLASSES = 2

def create_model(shape):
    input_layer=tf.keras.layers.Input(shape=shape)
    lstm_layer=tf.keras.layers.LSTM(LSTM_SIZE)(input_layer)
    dence_out=tf.keras.layers.Dense(1, 'sigmoid')(lstm_layer)

    return tf.keras.Model(input=input_layer, output=dence_out)

def dataset_processing(df):
    pass

df = pd.read_parquet(PATH + './dataset/DCS Бургер/hackaton2023_train.gzip')

df = dataset_processing(df)

target_weights = None
x_shape = None

model = create_model(x_shape)

optimizer = tf.keras.optimizers.Adam(1e-3, beta_1=0.9, beta_2=0.98,
                                     epsilon=1e-9)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(),
              loss_weights=target_weights,
              metrics=tfa.metrics.F1Score(num_classes=NUM_CLASSES, average='weighted'))

history = Model.fit(
    generator_dataset_train,
    epochs = EPOCHS,
    steps_per_epoch=int(train_size/BATCH_SIZE),
    validation_data=generator_dataset_test,
    validation_steps=int(test_size/BATCH_SIZE),
    )
