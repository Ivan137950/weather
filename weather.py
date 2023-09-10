import os

data_dir = '/Users/limon/OneDrive/Рабочий стол/GitHubProg/weather/jena_climate_2009_2016.csv'
fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')

with open(fname,'r', encoding='utf-8') as f:
    data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

# print(header,'\n')
# print(len(lines))
# print(lines[0], '\n')

#make a numpy
import numpy as np
float_data = np.zeros((len(lines), len(header) - 1))
for i, line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i, :] = values
# print(float_data[0])

# from matplotlib import pyplot as plt                     ///         WE CAN CHECK PLOT OF ANY PARAMETR
# temp = float_data[:, 1] # температура (в градусах Цельсия)
# plt.plot(range(len(temp)), temp)
# plt.show()

# LETS NORMOLIZE 
mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std
#Функция-генератор, возвращающая временные последовательности образцов и их целей
#lookback, delay   how much there were and there'll be intervals
def generator(data, lookback, delay, min_index, max_index, shuffle=False, batch_size=128, step=6):
#почему-то на этих строчках я вспомнил С...
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback, max_index, size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i += len(rows)
        samples = np.zeros((len(rows), lookback // step, data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples, targets

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
 lookback=lookback,
 delay=delay, min_index=0,
 max_index=200000,
 shuffle=True,batch_size=batch_size, step=step)

val_gen = generator(float_data,
 lookback=lookback,
 delay=delay,
 min_index=200001,
 max_index=300000,
 step=step,
 batch_size=batch_size)

test_gen = generator(float_data,
 lookback=lookback,
 delay=delay,
 min_index=300001,
 max_index=None,
 step=step,
 batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) // batch_size
test_steps = (len(float_data) - 300001 - lookback) // batch_size

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.GRU(32, input_shape=(None, float_data.shape[-1])))  #GRU ~ LSTM
model.add(layers.Dense(1))

model.compile(optimizer=RMSprop(), loss='mae')   #Absolute errrror
history = model.fit_generator(
 train_gen,
 steps_per_epoch=500,
 epochs=20,
 validation_data=val_gen,
 validation_steps=val_steps)

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)
import matplotlib.pyplot as plt

plt.figure()
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()
