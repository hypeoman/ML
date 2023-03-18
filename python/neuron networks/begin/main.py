import numpy as np
import tensorflow as tf

# Входные данные
x = np.array([])
y = np.array([])

XY = np.loadtxt('dataset.csv', delimiter=',')
print(XY)
print(XY)

print(XY[0][0])
print(XY[1][2])

for i in range(0, len(XY)):
    x = x + [XY[i][0], XY[i][1]]
    y = y + XY[i][2]

print(x)

print(y)

# Создание модели нейронной сети
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(x, y, epochs=1000)

# Предсказание пола по новым данным
new_data = np.array(
    [
    [164, 66],
    [176, 67],
    [185, 60],
    [155, 53]
    ]
)

predictions = model.predict(new_data)
print(predictions)

