import tensorflow as tf
import numpy as np

# Определение архитектуры нейросети
model = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(3, 3)), # Входной слой, плоский (3x3 = 9 нейронов)
  tf.keras.layers.Dense(64, activation='relu'), # Скрытый слой 1
  tf.keras.layers.Dense(64, activation='relu'), # Скрытый слой 2
  tf.keras.layers.Dense(9, activation='softmax') # Выходной слой (9 нейронов)
])

# Компиляция модели
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# 0 - ничего нет на поле
# 1 - крестик
# -1 - нолик

# Создаем игровую доску 3x3
board = np.zeros((3, 3), dtype=int)

# Задаем возможные комбинации для победы
winning_combinations = [[0, 1, 2], [3, 4, 5], [6, 7, 8], [0, 3, 6], [1, 4, 7], [2, 5, 8], [0, 4, 8], [2, 4, 6]]

# Создаем пустой список для хранения входных и выходных данных
dataset = []

# Создаем все возможные комбинации для игровой доски
for i in range(3):
    for j in range(3):
        # Создаем копию игровой доски
        temp_board = np.copy(board)
        # Задаем значение 1 на текущую позицию
        temp_board[i][j] = 1
        # Создаем все возможные комбинации для следующего хода
        for k in range(3):
            for l in range(3):
                # Проверяем, что позиция свободна
                if temp_board[k][l] == 0:
                    # Создаем копию игровой доски для следующего хода
                    next_board = np.copy(temp_board)
                    # Задаем значение -1 на следующую позицию
                    next_board[k][l] = -1
                    # Создаем список для входных и выходных данных
                    data = [temp_board, next_board]
                    # Добавляем список в датасет
                    dataset.append(data)

# Преобразуем датасет в массив Numpy
dataset = np.array(dataset)

# Входные данные для нейросети
X = dataset[:, 0]

# Выходные данные для нейросети
Y = dataset[:, 1]

# Преобразуем данные в формат, совместимый с нейросетью
X = X.reshape((X.shape[0], 3, 3))
Y = Y.reshape((Y.shape[0], 3, 3))

# Преобразуем значения из диапазона [-1, 1] в диапазон [0, 1]
X = (X + 1) / 2
Y = (Y + 1) / 2

# Конвертируем

model.fit(X, Y)