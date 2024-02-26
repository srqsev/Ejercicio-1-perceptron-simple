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

def plot_graph(weights, perceptron):
    if len(weights) == 3:
        for x1 in range(2):  # Range for OR
            for x2 in range(2):  # Range for OR
                inputs_with_bias = [1, x1, x2]  # Adding bias
                activation = perceptron.predict(inputs_with_bias)
                print(activation, end=' ')
            print()
    elif len(weights) == 2:
        for x1 in range(2):  # Range for XOR
            for x2 in range(2):  # Range for XOR
                inputs_with_bias = [1, x1, x2]  # Adding bias
                activation = perceptron.predict(inputs_with_bias)
                print(activation, end=' ')
            print()
    else:
        print("Invalid number of weights for plotting.")

def read_data(file_path):
    data = []
    with open(file_path, 'r') as file:
        for line in file:
            values = [float(x) for x in line.strip().split(',')]
            inputs = values[:-1]
            target = int(values[-1])
            data.append((inputs, target))
    return data



def main():
    # Cargar datos de entrenamiento para XOR
    training_data_xor = read_data("xor_trn.csv")
    num_inputs_xor = len(training_data_xor[0][0])
    learning_rate_xor = 0.1
    max_epochs_xor = 100

    # Crear y entrenar el perceptrón para XOR
    perceptron_xor = Perceptron(num_inputs_xor, learning_rate_xor, max_epochs_xor)
    perceptron_xor.train(training_data_xor)

    print("Pesos finales para XOR:", perceptron_xor.weights)

    # Cargar datos de prueba para XOR
    test_data_xor = read_data("xor_tst.csv")

    # Prueba de XOR
    print("Prueba de XOR:")
    for inputs, target in test_data_xor:
        prediction = perceptron_xor.predict([1] + inputs)
        print(f"Inputs: {inputs}, Target: {target}, Prediction: {prediction}")

    # Cargar datos de entrenamiento para OR
    training_data_or = read_data("or_trn.csv")
    num_inputs_or = len(training_data_or[0][0])
    learning_rate_or = 0.1
    max_epochs_or = 100

    # Crear y entrenar el perceptrón para OR
    perceptron_or = Perceptron(num_inputs_or, learning_rate_or, max_epochs_or)
    perceptron_or.train(training_data_or)

    print("Pesos finales para OR:", perceptron_or.weights)

    # Cargar datos de prueba para OR
    test_data_or = read_data("or_tst.csv")

    # Prueba de OR
    print("Prueba de OR:")
    for inputs, target in test_data_or:
        prediction = perceptron_or.predict([1] + inputs)
        print(f"Inputs: {inputs}, Target: {target}, Prediction: {prediction}")

    # Dibujar la gráfica en la consola para XOR
    print("Gráfica de entrenamiento para XOR:")
    plot_graph(perceptron_xor.weights, perceptron_xor)

    # Dibujar la gráfica en la consola para OR
    print("Gráfica de entrenamiento para OR:")
    plot_graph(perceptron_or.weights, perceptron_or)

if __name__ == "__main__":
    main()
