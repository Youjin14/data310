import tensorflow as tf
import numpy as np
from tensorflow import keras
model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')
xs = np.array([4.0, 3.0, 5.0, 4.0, 2.0, 3.0], dtype=float)
ys = np.array([399, 97, 347.5, 289, 250, 229], dtype=float)
model.fit(xs, ys, epochs=500)
print(model.predict([1.0,2.0,3.0,4.0,5.0]))
