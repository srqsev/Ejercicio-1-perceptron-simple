class Perceptron:
    def __init__(self, num_inputs, learning_rate=0.1, max_epochs=100):
        self.num_inputs = num_inputs
        self.learning_rate = learning_rate
        self.max_epochs = max_epochs
        self.weights = [0] * (num_inputs + 1)  # +1 for bias

    def train(self, training_data):
        epoch = 0
        while epoch < self.max_epochs:
            error_count = 0
            for inputs, target in training_data:
                inputs_with_bias = [1] + inputs  # Adding bias
                activation = self.predict(inputs_with_bias)
                error = target - activation
                if error != 0:
                    error_count += 1
                    for i in range(self.num_inputs + 1):
                        self.weights[i] += self.learning_rate * error * inputs_with_bias[i]
            if error_count == 0:
                break
            epoch += 1

    def predict(self, inputs):
        activation = 0
        for i in range(self.num_inputs + 1):
            activation += self.weights[i] * inputs[i]
        return 1 if activation >= 0 else 0

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = [float(x) for x in line.strip().split(',')]
            inputs = values[:-1]
            target = int(values[-1])
            data.append((inputs, target))
    return data


def plot_graph(weights):
    for y in range(-10, 11):
        row = ''
        for x in range(-10, 11):
            activation = weights[0] + weights[1] * x + weights[1] * y  # Solo dos pesos
            if activation >= 0:
                row += 'X'
            else:
                row += 'O'
        print(row)








def main():
    # Leer datos de entrenamiento
    training_data = read_data("XOR_trn.csv")
    num_inputs = len(training_data[0]) - 1
    learning_rate = 0.1
    max_epochs = 100

    # Crear y entrenar el perceptrón
    perceptron = Perceptron(num_inputs, learning_rate, max_epochs)
    perceptron.train(training_data)

    print("Pesos finales:", perceptron.weights)

    # Dibujar la gráfica en la consola
    print("Gráfica de entrenamiento:")
    plot_graph(perceptron.weights)

    # Leer datos de prueba
    test_data = read_data("XOR_tst.csv")

    # Prueba del perceptrón entrenado con datos de prueba
    print("Resultados de la prueba:")
    for inputs, target in test_data:
        inputs_with_bias = [1] + inputs
        prediction = perceptron.predict(inputs_with_bias)
        print("Entradas:", inputs, "Target:", target, "Predicción:", prediction)

if __name__ == "__main__":
    main()
