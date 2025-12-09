import numpy as np
from random import shuffle


class Layer:
    def __init__(self, input, output, bias=False, activation=None):
        self.__weights = np.random.rand(input, output)
        self.__alpha = 1e-3  # Default value

        self.__bias = bias
        if bias:
            self.__weights = np.vstack((self.__weights, np.ones((1, self.__weights.shape[1]))))

        self.__activation_layer = activation
        self.__pre_activated_output = None
        self.__curr_inputs = None
        self.__update = None

    def __call__(self, input):
        """Forward pass through layer

        Args:
            input (np.array): layer inputs

        Returns:
            np.array: layer outputs
        """
        # Handle optional bias term
        if self.__bias:
            input = np.hstack((input, np.ones((input.shape[0], 1))))

        self.__curr_inputs = np.copy(input)
        weight_prod = input @ self.__weights

        # Handle optional activation layer
        if self.__activation_layer:
            self.__pre_activated_output = np.copy(weight_prod)
            weight_prod = self.__activation_layer(weight_prod)

        return weight_prod

    def back(self, ret):
        """Backward pass through layer

        Args:
            ret (np.array): gradients calculated till next layer

        Returns:
            np.array: gradients calculated till current layer
        """
        if self.__activation_layer:  # Optional activation layer
            ret = self.__activation_layer.derivative(self.__pre_activated_output, ret)

        self.__update = (self.__curr_inputs.T) @ ret
        new_ret = ret @ (self.__weights.T)

        if self.__bias:  # Remove bias column if needed
            new_ret = new_ret[:, :-1]

        return new_ret

    def update(self):
        """Update layer weights
        """
        self.__weights -= self.__alpha * self.__update
        self.__pre_activated_output = None
        self.__curr_inputs = None
        self.__update = None

    def update_alpha(self, new_alpha):
        """Learning rate setter method

        Args:
            new_alpha (float): new alpha value
        """
        self.__alpha = new_alpha


class LayerList:
    def __init__(self, *layers):
        if len(layers) == 0:
            self.layer_list = list()
        else:
            self.layer_list = list(layers)

    def append(self, *layers):
        """Add layers to the model outside the initialization
        """
        for layer in layers:
            self.layer_list.append(layer)

    def __call__(self, input):
        """Forward pass through model

        Args:
            input (np.array): input data

        Returns:
            np.array: model predictions
        """
        for layer in self.layer_list:
            input = layer(input)

        return input

    def back(self, error):
        """Backward pass through model

        Args:
            error (np.array): the gradients calculated by the loss function
        """
        for layer in self.layer_list[::-1]:
            error = layer.back(error)

    def step(self):
        """Update all weights and biases in the model
        """
        for layer in self.layer_list:
            layer.update()

    def predict(self, inputs):
        """Model inference function

        Args:
            inputs (np.array): inference inputs

        Returns:
            np.array: model predictions
        """
        predictions = []

        for input in inputs:
            # Expanding dims to ensure shape is (1, num_features)
            predictions.append(self(np.expand_dims(input, axis=0)))

        return predictions

    @staticmethod
    def batch(input_data, expected, batch_size):
        """Split the training dataset into batches

        Args:
            input_data (np.array): input dataset
            expected (np.array): expected values for inputs
            batch_size (int): desired batch size

        Returns:
            _type_: _description_
        """
        data_pts = input_data.shape[0]
        indicies = [i for i in range(data_pts)]
        shuffle(indicies)
        batched_data, batched_results = [], []

        for i in range(data_pts // batch_size):
            data_batch, expected_batch = [], []

            for j in range(batch_size):
                data_batch.append(input_data[i * batch_size + j])
                expected_batch.append(expected[i * batch_size + j])

            batched_data.append(data_batch)
            batched_results.append(expected_batch)

        return np.array(batched_data), np.array(batched_results)

    def update_alpha(self, new_alpha):
        """Update the learning rate in all layers

        Args:
            new_alpha (float): updated alpha value
        """
        for layer in self.layer_list:
            layer.update_alpha(new_alpha)

    def fit(self, input_data, expected, epochs, alpha, batch_size, loss_deriv_func):
        """Model training loop

        Args:
            input_data (np.array): model training data
            expected (np.array): expected values for training data
            epochs (int): number of times the input_data is fed to the model
            alpha (float): initial learning rate
            batch_size (int): batch size for training
            loss_deriv_func (function): loss function (from loss.py)
        """
        if len(self.layer_list) == 0:
            return

        total_iter = epochs
        self.update_alpha(alpha)

        while epochs:
            epochs -= 1
            batched_data, batched_expected = LayerList.batch(input_data, expected, batch_size)

            for idx, data_batch in enumerate(batched_data):
                output = self(data_batch)
                self.back(loss_deriv_func(output, batched_expected[idx]))
                self.step()

            if epochs == total_iter // 10:
                # Reducing learning rate to hone in on minima of loss function
                alpha /= 10
                self.update_alpha(alpha)