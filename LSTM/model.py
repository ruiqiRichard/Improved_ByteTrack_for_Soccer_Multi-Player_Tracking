import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense

class LSTMTracker:
    """
    An LSTM-based model for predicting object trajectories.

    The input consists of a sequence of bounding box coordinates in the format:
        x, y, a, h
    
    Where:
        - (x, y) is the center position
        - a is the aspect ratio
        - h is the height of the bounding box
    
    """

    def __init__(self, input_dim=4, output_dim=4, seq_length=10, lstm_units=64):
        """
        Initialize the LSTM model.

        Parameters:
        - input_dim: int, number of features in each input vector (default is 4 for [x, y, a, h]).
        - output_dim: int, number of features in the output vector (default is 4).
        - seq_length: int, length of input sequences.
        - lstm_units: int, number of LSTM units.
        """
        self.seq_length = seq_length

        # Define the LSTM model
        self.model = Sequential([
            LSTM(lstm_units, input_shape=(seq_length, input_dim), return_sequences=True),
            Dense(output_dim)
        ])
        
        self.model.compile(optimizer='adam', loss='mse')

    def train(self, X_train, y_train, epochs=50, batch_size=32):
        """
        Train the LSTM model.

        Parameters:
        - X_train: ndarray, input training data of shape (num_samples, seq_length, input_dim).
        - y_train: ndarray, target data of shape (num_samples, output_dim).
        - epochs: int, number of training epochs.
        - batch_size: int, size of training batches.
        """
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=1)

    def predict(self, input_sequence):
        """
        Predict the next state given an input sequence.

        Parameters:
        - input_sequence: ndarray, input sequence of shape (seq_length, input_dim).

        Returns:
        - ndarray, predicted state of shape (output_dim,).
        """
        input_sequence = np.expand_dims(input_sequence, axis=0)
        return self.model.predict(input_sequence, verbose=0)[0]

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        Parameters:
        - file_path: str, path to save the model.
        """
        self.model.save(file_path)

    def load_model(self, file_path):
        """
        Load a trained model from a file.

        Parameters:
        - file_path: str, path to the model file.
        """
        self.model = tf.keras.models.load_model(file_path)


if __name__ == "__main__":
    # Generate synthetic data for demonstration
    num_samples = 1000
    seq_length = 10
    input_dim = 4

    X_train = np.random.rand(num_samples, seq_length, input_dim)
    y_train = np.random.rand(num_samples, input_dim)

    # Initialize and train the LSTM tracker
    tracker = LSTMTracker(input_dim=input_dim, seq_length=seq_length)
    tracker.train(X_train, y_train, epochs=10, batch_size=32)

    # Predict the next state for a sample sequence
    sample_sequence = X_train[0]
    prediction = tracker.predict(sample_sequence)
    print("Predicted state:", prediction)
