import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Flatten
import logging


def generate_mlp(num_layers:int, initial_neurons:int, output_classes:int, dropout_rate:float, activation_function:str, scaling_factor:float):
    '''scaling factor - between 0 and 1.'''
    model = tf.keras.Sequential()

    for i in range(num_layers):
        # For the first layer, add the specified number of neurons
        if i == 0:
            model.add(Dense(initial_neurons, activation=activation_function))
        else:
            # Decrease the number of neurons by the scaling factor
            neurons = int(initial_neurons * (scaling_factor ** i))
            model.add(Dense(neurons, activation=activation_function))

        # Add dropout after each Dense layer
        model.add(Dropout(dropout_rate))

    # Add the final layer with output_classes neurons
    model.add(Dense(output_classes, activation='softmax'))
    logging.debug("MLP Layer Completed.")


    return model

# Example usage:

if __name__ == "__main__":
    mlp_model = generate_mlp(num_layers=5, initial_neurons=512, output_classes=3, dropout_rate=0.5, activation_function='relu', scaling_factor=.5)
