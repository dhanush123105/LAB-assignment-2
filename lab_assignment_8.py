import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler


# A1

def compute_net(input_vec, weight_vec, bias_val):
    return np.dot(input_vec, weight_vec) + bias_val

def step_activation(x):
    return 1 if x >= 0 else 0

def bipolar_activation(x):
    return 1 if x >= 0 else -1

def sigmoid_activation(x):
    return 1 / (1 + np.exp(-x))

def relu_activation(x):
    return max(0, x)

def tanh_activation(x):
    return np.tanh(x)

def leaky_relu_activation(x):
    return x if x > 0 else 0.01 * x

def compute_error(target_val, predicted_val):
    return target_val - predicted_val


def train_model(features, labels, learning_rate, activation_fn, max_epochs=1000):
    sample_count, feature_count = features.shape
    weights = np.random.randn(feature_count)
    bias = np.random.randn()

    epoch_errors = []

    for epoch in range(max_epochs):
        total_error = 0

        for idx in range(sample_count):
            net_val = compute_net(features[idx], weights, bias)
            output_val = activation_fn(net_val)
            err = compute_error(labels[idx], output_val)

            weights += learning_rate * err * features[idx]
            bias += learning_rate * err
            total_error += err ** 2

        mse = total_error / sample_count
        epoch_errors.append(mse)

        if mse <= 0.002:
            break

    return weights, bias, epoch_errors


def make_predictions(features, weights, bias, activation_fn):
    return np.array([activation_fn(compute_net(sample, weights, bias)) for sample in features])


# A2

def get_and_data():
    input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    target_data = np.array([0,0,0,1])
    return input_data, target_data


# A5

def get_xor_data():
    input_data = np.array([[0,0],[0,1],[1,0],[1,1]])
    target_data = np.array([0,1,1,0])
    return input_data, target_data

# A4

def evaluate_learning_rates(features, labels):
    lr_values = np.arange(0.1, 1.1, 0.1)
    epoch_counts = []

    for lr in lr_values:
        _, _, error_list = train_model(features, labels, lr, step_activation)
        epoch_counts.append(len(error_list))

    return lr_values, epoch_counts

# A6

def get_customer_dataset():
    dataset = np.array([
        [20,6,2,386,1],
        [16,3,6,289,1],
        [27,6,2,393,1],
        [19,1,2,110,0],
        [24,4,2,280,1],
        [22,1,5,167,0],
        [15,4,2,271,1],
        [18,4,2,274,1],
        [21,1,4,148,0],
        [16,2,4,198,0]
    ])

    feature_matrix = dataset[:, :-1]
    label_vector = dataset[:, -1]

    feature_matrix = (feature_matrix - feature_matrix.mean(axis=0)) / feature_matrix.std(axis=0)
    return feature_matrix, label_vector



# A7


def compute_pseudo_inverse(features, labels):
    bias_added = np.c_[np.ones(features.shape[0]), features]
    weights_pi = np.linalg.pinv(bias_added).dot(labels)
    return weights_pi



# A8


def train_backpropagation(lr=0.05, max_epochs=1000):
    inputs, targets = get_and_data()

    hidden_weights = np.random.randn(2,2)
    output_weights = np.random.randn(2,1)

    error_history = []

    for epoch in range(max_epochs):
        total_error = 0

        for i in range(len(inputs)):
            sample = inputs[i].reshape(1,-1)
            target = targets[i]

            hidden_out = sigmoid_activation(np.dot(sample, hidden_weights))
            final_out = sigmoid_activation(np.dot(hidden_out, output_weights))

            err = target - final_out
            total_error += err**2

            delta_out = err * final_out * (1 - final_out)
            delta_hidden = hidden_out * (1 - hidden_out) * np.dot(delta_out, output_weights.T)

            output_weights += lr * hidden_out.T.dot(delta_out)
            hidden_weights += lr * sample.T.dot(delta_hidden)

        mse = total_error / len(inputs)
        error_history.append(mse)

        if mse <= 0.002:
            break

    return error_history



# A10


def get_two_output_and():
    inputs, targets = get_and_data()
    encoded_targets = np.array([[1,0] if val==0 else [0,1] for val in targets])
    return inputs, encoded_targets



# A12


def load_parkinsons_data(file_path):
    dataframe = pd.read_csv(file_path)

    if 'name' in dataframe.columns:
        dataframe = dataframe.drop(columns=['name'])

    labels = dataframe['status'].values
    features = dataframe.drop(columns=['status']).values

    scaler = StandardScaler()
    features = scaler.fit_transform(features)

    return features, labels



# Main


if __name__ == "__main__":

    # AND Gate
    and_inputs, and_targets = get_and_data()
    weights, bias, error_list = train_model(and_inputs, and_targets, 0.05, step_activation)
    print("A2 AND Gate Epochs:", len(error_list))

    # Activation comparison
    for act_fn in [bipolar_activation, sigmoid_activation, relu_activation]:
        _, _, err_vals = train_model(and_inputs, and_targets, 0.05, act_fn)
        print("Activation:", act_fn.__name__, "Epochs:", len(err_vals))

    # Learning rate
    lr_vals, iter_counts = evaluate_learning_rates(and_inputs, and_targets)
    plt.plot(lr_vals, iter_counts)
    plt.xlabel("Learning Rate")
    plt.ylabel("Iterations")
    plt.title("Learning Rate vs Iterations")
    plt.show()

    # XOR
    xor_inputs, xor_targets = get_xor_data()
    _, _, xor_errors = train_model(xor_inputs, xor_targets, 0.05, step_activation)
    print("A5 XOR Epochs:", len(xor_errors))

    # Customer data
    cust_X, cust_y = get_customer_dataset()
    weights, bias, _ = train_model(cust_X, cust_y, 0.05, sigmoid_activation)
    predictions = make_predictions(cust_X, weights, bias, sigmoid_activation)
    print("A6 Accuracy:", np.mean((predictions > 0.5) == cust_y))

    # Pseudo inverse
    weights_pi = compute_pseudo_inverse(cust_X, cust_y)
    print("A7 Pseudo-inverse weights:", weights_pi)

    # Backprop
    bp_errors = train_backpropagation()
    print("A8 Backprop epochs:", len(bp_errors))

    # Two output
    two_X, two_y = get_two_output_and()
    print("A10 Sample Output:", two_y[:3])

    # MLP
    model = MLPClassifier(hidden_layer_sizes=(2,), max_iter=1000)
    model.fit(and_inputs, and_targets)
    print("A11 AND Accuracy:", model.score(and_inputs, and_targets))

    model.fit(xor_inputs, xor_targets)
    print("A11 XOR Accuracy:", model.score(xor_inputs, xor_targets))

    # Parkinson
    park_X, park_y = load_parkinsons_data("parkinsons.csv")
    model.fit(park_X, park_y)
    print("A12 Parkinson Accuracy:", model.score(park_X, park_y))