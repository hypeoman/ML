import numpy as np

def sigmoid(x):
  # Наша функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))

def mse_loss(y_true, y_pred):
  # y_true и y_pred - массивы numpy одинаковой длины.
  return ((y_true - y_pred) ** 2).mean()

class Neuron:
    def __init__(self, weights, bias):
        self.weights = weights
        self.bias = bias

    def feedforward(self, inputs):
        # 
        total = np.dot(self.weights, inputs) + self.bias
        return sigmoid(total)


class OurNeuralNetwork:
  '''
  Нейронная сеть с:
    - 2 входами
    - скрытым слоем с 2 нейронами (h1, h2)
    - выходным слоем с 1 нейроном (o1)
  Все нейроны имеют одинаковые веса и пороги:
    - w = [0, 1]
    - b = 0
  '''
  def __init__(self):
    weights = np.array([-2.5, 1.5])
    bias = 0

    # Используем класс Neuron из предыдущего раздела
    self.h1 = Neuron(weights, bias)
    self.h2 = Neuron(weights, bias)
    self.o1 = Neuron(weights, bias)

  def feedforward(self, x):
    out_h1 = self.h1.feedforward(x)
    out_h2 = self.h2.feedforward(x)

    # Входы для o1 - это выходы h1 и h2
    out_o1 = self.o1.feedforward(np.array([out_h1, out_h2]))

    return out_o1

network = OurNeuralNetwork()
x = np.array([
    [133, 65], # Алиса
    [160, 72], # Боб
    [152, 70], # Чарли
    [120, 60]  # Диана
    ])

# 1 - женщина, 0 - мужчина
y_true = np.array([
    1, # Алиса
    0, # Боб
    0, # Чарли
    1  # Диана
    ])

n = 4

y_pred = np.array([])

for i in range(0,4):
    network = OurNeuralNetwork()
    xn = np.array([x[i][0], x[i][1]])
    output = network.feedforward(xn)
    # print(output)
    y_pred = np.append(y_pred, output)

print("MSE_LOSS =",mse_loss(y_true, y_pred))