import os

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow_addons.optimizers import LAMB

PATH = os.getcwd()
FILE_NAME = './END_minmax_rev.parquet'
TEST_FILE_NAME = './test.parquet'
HIDDEN_SIZE = 128
NUM_CLASSES = 1

EPOCHS = 10
BATCH_SIZE = 1024
TEST_SPLIT = 0.35

NUM_MODELS = 3
INITIALIZER = tf.keras.initializers.LecunUniform
REGUL = tf.keras.regularizers.L2
REGUL_P = 0.001

def create_model(shape):
    input_layer=tf.keras.layers.Input(shape=shape)
    input_layer = tf.keras.layers.BatchNormalization()(input_layer)

    hidden_layer=tf.keras.layers.Dense(HIDDEN_SIZE,
                                     activation=None,
                                     kernel_initializer=INITIALIZER(),
                                     kernel_regularizer=REGUL(REGUL_P)
                                     )(input_layer)
    hidden_layer=tf.keras.layers.LeakyReLU()(hidden_layer)
 
    hidden_layer=tf.keras.layers.Dense(HIDDEN_SIZE,
                                     activation=None,
                                     kernel_initializer=INITIALIZER(),
                                     kernel_regularizer=REGUL(REGUL_P)
                                     )(hidden_layer)
    hidden_layer=tf.keras.layers.LeakyReLU()(hidden_layer)
 
    dence_out=tf.keras.layers.Dense(1,
                                    activation='sigmoid',
                                    kernel_initializer=INITIALIZER())(hidden_layer)
    return tf.keras.Model(inputs=input_layer, outputs=dence_out)

# Загрузим датасет, сбалансируем, преобразуем в numpy
df = pd.read_parquet(PATH + FILE_NAME)
for col in df.columns:
    if df[col].dtype == object:
        df[col] = df[col].astype('category').cat.codes

df = df.groupby('buy_post')
df = pd.DataFrame(df.apply(lambda x: x.sample(df.size().min()).reset_index(drop=True)))

df =  df.iloc[np.random.permutation(len(df))]
df =  df.iloc[np.random.permutation(len(df))]


y = df['buy_post']
target_weights = df['buy_post'].value_counts().map(lambda x: 1- x/len(df)).to_dict()

x = df.drop(['buy_post', 'date_diff_post',], axis=1).to_numpy(na_value=0)#.astype('float')

# Закодируем данные, отнормализуем
x = np.log1p(np.abs(x))
x -= x.mean()
x /= x.std()
# x = 0.1 + (x-x.min())/(x.max()-x.min())*0.9

x_shape = (*x.shape[1:], )

# Здесь выбирается либо ансамбль, либо одна модель
if NUM_MODELS == 1:
    model = create_model(x_shape)

    
else:
    models = [create_model(x_shape) for _ in range(NUM_MODELS)]
    m_input = tf.keras.layers.Input(shape = x_shape)
    m_output = [model(m_input) for model in models]
    ensemble_output = tf.keras.layers.Concatenate()(m_output)
    ensemble_output = tf.keras.layers.Dense(HIDDEN_SIZE,
                                     activation=None,
                                     kernel_initializer=INITIALIZER(),
                                     kernel_regularizer=REGUL(REGUL_P))(ensemble_output)
    ensemble_output=tf.keras.layers.LeakyReLU()(ensemble_output)
    ensemble_output = tf.keras.layers.Dropout(0.5)(ensemble_output)
    ensemble_output = tf.keras.layers.Dense(1, activation=None)(ensemble_output)
    model = tf.keras.Model(inputs=m_input, outputs=ensemble_output)

# Скомпилируем модель
optimizer=tf.keras.optimizers.Adam(
            learning_rate=tf.keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=2e-3,
                decay_steps=3000,
                decay_rate=0.95,
            )
    )
model.compile(optimizer=optimizer,
            loss= tf.keras.losses.BinaryCrossentropy(),
            metrics=[
                tfa.metrics.F1Score(num_classes=NUM_CLASSES, threshold=0.5, average='macro'),
                     tf.keras.metrics.AUC(),
                     'acc', 
                     ] )


# Обучим модель
history = model.fit(
    x, y,
    epochs = EPOCHS,
    batch_size = BATCH_SIZE,
    validation_split = TEST_SPLIT,
    class_weight=target_weights
    )

# Загрузка и обработка тестового датасета
submit_df = pd.read_parquet(PATH + TEST_FILE_NAME)
for col in submit_df.columns:
    if submit_df[col].dtype == object:
        submit_df[col] = submit_df[col].astype('category').cat.codes
submit_x = submit_df.to_numpy(na_value=0)
submit_x = np.log1p(np.abs(submit_x))
submit_x -= submit_x.mean()
submit_x /= submit_x.std()
submit_x = 0.1 + (submit_x-submit_x.min())/(submit_x.max()-submit_x.min())*0.9

# Сохранение сабмитов
result = model.predict(submit_x)
sub = pd.read_csv(PATH + './submission.csv',sep=";")
sub.customer_id = pd.read_csv(PATH + './new_f3.csv',sep=";").customer_id
sub['buy_post'] = np.where(result > 0.5, 1, 0)
sub.to_csv("nn_most_common_2.csv",index=False,sep=';')