import numpy as np
import tensorflow as tf

# Входные данные
X = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0, -1, 0, 0],
        [1, 1, 0, 0, 1, 0, -1, 0, -1]
        # [160, 55],
        # [165, 65],
        # [182, 85]
    ]
)
y = np.array(
    [31,
     33,
     32
     # 1,
     # 1,
     # 0
     ]
)

# Создание модели нейронной сети
model = tf.keras.models.Sequential([
  tf.keras.layers.Dense(64, activation='relu', input_shape=(9,)), # Входящий слой нейроннов
  tf.keras.layers.Dense(64, activation='relu'), # Внешний слов
  tf.keras.layers.Dense(1, activation='softmax') # Выходящий слой
])

# Компиляция модели
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Обучение модели
model.fit(X, y, epochs=1000)

# Предсказание пола по новым данным
new_data = np.array(
    [
        [1, 1, 0, -1, 1, 0, -1, 0, -1]
    ]
)

predictions = model.predict(new_data)
print(predictions)