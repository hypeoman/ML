import numpy as np
import tensorflow as tf

# Входные данные
X = np.array([[170, 70], [175, 73], [180, 80], [160, 55], [165, 65], [182, 85]])
y = np.array([0, 0, 0, 1, 1, 0]) # 0 - женщины, 1 - мужчины

# Создание модели нейронной сети
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(2,)),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, epochs=1000)

# Предсказание пола по новым данным
new_data = np.array(
    [
    [164, 66],
    [176, 67],
    [185, 60]
    ]
)

predictions = model.predict(new_data)
print(predictions)