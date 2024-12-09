import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import pickle
import matplotlib.pyplot as plt

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from dataset import DataLoader

class LSTMTracker:
    """
    An LSTM-based model for predicting object trajectories.
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
        
    def train_with_dataloader(self, data_loader, num_epochs=50, batch_size=32):
        """
        Train the LSTM model using the provided DataLoader.

        Parameters:
        - data_loader: DataLoader instance, handles loading of training data.
        - num_epochs: int, number of training epochs.
        - batch_size: int, size of training batches.
        """
        loss_history = []
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)

        for epoch in range(num_epochs):
            data_loader.reset_batch_pointer()
            epoch_loss = 0  # To track loss for each epoch

            for batch in range(data_loader.num_batches):
                x_batch, y_batch = data_loader.next_batch()

                x_batch = np.array(x_batch).reshape((batch_size, self.seq_length, -1))
                y_batch = np.array(y_batch)

                with tf.GradientTape() as tape:
                    predictions = self.model(x_batch)
                    loss = tf.reduce_mean(tf.square(predictions - y_batch))

                gradients = tape.gradient(loss, self.model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

                loss_value = loss.numpy()
                loss_history.append(loss_value)
                epoch_loss += loss_value

                print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch + 1}/{data_loader.num_batches}, Loss: {loss_value:.4f}")

            # Average loss for the epoch
            avg_epoch_loss = epoch_loss / data_loader.num_batches
            print(f"Epoch {epoch + 1} Average Loss: {avg_epoch_loss:.4f}")

        # Plot training loss
        plt.plot(loss_history, label="Training Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss Per Batch")
        plt.legend()
        plt.show()

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

    def save_predictions(self, data_loader, file_path):
        """
        Save predictions to a file.

        Parameters:
        - data_loader: DataLoader instance.
        - file_path: str, path to save the predictions.
        """
        input_data = data_loader.data
        input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(
            input_data, padding='post', maxlen=self.seq_length, dtype='float32'
        )

        predictions = self.model.predict(input_data_padded)

        pred_results = {
            "predictions": predictions,
            "ground_truth": input_data_padded
        }

        with open(file_path, "wb") as f:
            pickle.dump(pred_results, f)


if __name__ == "__main__":
    # Training configuration
    input_dim = 4
    output_dim = 4
    seq_length = 10
    batch_size = 50
    num_epochs = 20

    # Initialize data loader
    data_loader = DataLoader(batch_size=batch_size, seq_length=seq_length)

    # Initialize and train the tracker
    tracker = LSTMTracker(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length)
    tracker.train_with_dataloader(data_loader, num_epochs=num_epochs, batch_size=batch_size)

    # Save predictions
    tracker.save_predictions(data_loader, "pred_results.pkl")
