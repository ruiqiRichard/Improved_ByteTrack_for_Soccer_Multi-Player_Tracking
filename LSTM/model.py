import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import pickle
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from dataset import DataLoader

class LSTMTracker(nn.Module):
    """
    An LSTM-based model for predicting object trajectories in PyTorch.
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
        super(LSTMTracker, self).__init__()
        self.seq_length = seq_length
        self.img_width = 1920
        self.img_height = 1080

        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, output_dim)

    def forward(self, x):
        """
        Forward pass for the LSTMTracker.

        Parameters:
        - x: torch.Tensor, input tensor of shape (batch_size, seq_length, input_dim).

        Returns:
        - torch.Tensor, output tensor of shape (batch_size, seq_length, output_dim).
        """
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)

    def train_with_dataloader(self, train_loader, test_loader, num_epochs=50, batch_size=32, learning_rate=0.001):
        """
        Train the LSTM model using the provided DataLoader.

        Parameters:
        - train_loader: DataLoader instance for training.
        - test_loader: DataLoader instance for validation.
        - num_epochs: int, number of training epochs.
        - batch_size: int, size of training batches.
        - learning_rate: float, learning rate for the optimizer.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        criterion = nn.MSELoss()

        for epoch in range(num_epochs):
            self.train()
            train_loader.reset_batch_pointer()
            epoch_loss = 0

            for batch in tqdm(range(train_loader.num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
                x_batch, y_batch = train_loader.next_batch()
                x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)

                optimizer.zero_grad()
                predictions = self(x_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_epoch_loss = epoch_loss / train_loader.num_batches
            print(f"Epoch {epoch + 1} Average Train Loss: {avg_epoch_loss:.7f}")

            # Evaluate on validation data
            self.evaluate(test_loader)

    def evaluate(self, data_loader):
        """
        Evaluate the model on the provided DataLoader.

        Parameters:
        - data_loader: DataLoader instance for validation.
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

        data_loader.reset_batch_pointer()
        total_loss = 0
        criterion = nn.MSELoss()

        with torch.no_grad():
            for batch in tqdm(range(data_loader.num_batches)):
                x_batch, y_batch = data_loader.next_batch()
                x_batch = torch.tensor(x_batch, dtype=torch.float32).to(device)
                y_batch = torch.tensor(y_batch, dtype=torch.float32).to(device)

                predictions = self(x_batch)
                loss = criterion(predictions, y_batch)
                total_loss += loss.item()

        avg_loss = total_loss / data_loader.num_batches
        print(f"Average Validation Loss: {avg_loss:.7f}")

    def predict(self, input_sequence):
        """
        Predict the next state given an input sequence.

        Parameters:
        - input_sequence: ndarray, input sequence of shape (seq_length, input_dim).

        Returns:
        - torch.Tensor, predicted state of shape (output_dim,).
        """
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.to(device)
        self.eval()

        input_sequence = torch.tensor(input_sequence, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            predictions = self(input_sequence).squeeze(0)
        predictions = predictions.cpu().numpy()
        predictions[:, 0] *= self.img_width
        predictions[:, 1] *= self.img_height
        predictions[:, 2] *= self.img_width
        predictions[:, 3] *= self.img_height
        return predictions

    def save_model(self, file_path):
        """
        Save the trained model to a file.

        Parameters:
        - file_path: str, path to save the model.
        """
        torch.save(self.state_dict(), file_path)

    def load_model(self, file_path):
        """
        Load a trained model from a file.

        Parameters:
        - file_path: str, path to load the model.
        """
        self.load_state_dict(torch.load(file_path))

    def save_predictions(self, data_loader, file_path):
        """
        Save predictions to a file.

        Parameters:
        - data_loader: DataLoader instance.
        - file_path: str, path to save the predictions.
        """
        input_data = data_loader.data
        input_data_padded = np.pad(input_data, ((0, 0), (0, self.seq_length - input_data.shape[1]), (0, 0)), mode='constant')

        predictions = []
        with torch.no_grad():
            for seq in input_data_padded:
                seq = torch.tensor(seq, dtype=torch.float32).unsqueeze(0)
                preds = self(seq).squeeze(0).numpy()
                predictions.append(preds)

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
    batch_size = 64
    num_epochs = 10

    # Initialize data loader
    train_loader = DataLoader(batch_size=batch_size, seq_length=seq_length)
    test_loader = DataLoader(batch_size=batch_size, seq_length=seq_length, train=False)

    # Initialize and train the tracker
    tracker = LSTMTracker(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length)
    tracker.train_with_dataloader(train_loader, test_loader, num_epochs=num_epochs, batch_size=batch_size)

    # Save the model
    tracker.save_model("lstm_model.pth")
    # tracker.save_model("lstm_model.keras")  # Reference to the original Keras save line

    # Evaluate the model
    tracker.evaluate(test_loader)


# import numpy as np
# import tensorflow as tf
# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import LSTM, Dense
# import pickle
# import matplotlib.pyplot as plt
# from tqdm import tqdm
# import sys
# import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# from dataset import DataLoader

# # @tf.keras.saving.register_keras_serializable()
# class LSTMTracker:
#     """
#     An LSTM-based model for predicting object trajectories.
#     """

#     def __init__(self, input_dim=4, output_dim=4, seq_length=10, lstm_units=64):
#         """
#         Initialize the LSTM model.

#         Parameters:
#         - input_dim: int, number of features in each input vector (default is 4 for [x, y, a, h]).
#         - output_dim: int, number of features in the output vector (default is 4).
#         - seq_length: int, length of input sequences.
#         - lstm_units: int, number of LSTM units.
#         """
#         self.seq_length = seq_length
#         self.img_width = 1920
#         self.img_height = 1080
#         # Define the LSTM model
#         self.model = Sequential([
#             LSTM(lstm_units, input_shape=(seq_length, input_dim), return_sequences=True),
#             Dense(output_dim)
#         ])
        
#         self.model.compile(optimizer='adam', loss='mse')
        
#     def train_with_dataloader(self, data_loader, test_dataloader, num_epochs=50, batch_size=32):
#         """
#         Train the LSTM model using the provided DataLoader.

#         Parameters:
#         - data_loader: DataLoader instance, handles loading of training data.
#         - num_epochs: int, number of training epochs.
#         - batch_size: int, size of training batches.
#         """
#         loss_history = []
#         optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

#         for epoch in (range(num_epochs)):
#             data_loader.reset_batch_pointer()
#             epoch_loss = 0  # To track loss for each epoch

#             for batch in tqdm(range(data_loader.num_batches), desc=f"Epoch {epoch + 1}/{num_epochs}"):
#                 x_batch, y_batch = data_loader.next_batch()

#                 x_batch = np.array(x_batch).reshape((batch_size, self.seq_length, -1))
#                 y_batch = np.array(y_batch)
#                 # print(y_batch)
                
#                 # print(x_batch.shape, y_batch.shape)

#                 with tf.GradientTape() as tape:
#                     predictions = self.model(x_batch)
#                     # predictions[:, :, 0] *= self.img_width
#                     # predictions[:, :, 1] *= self.img_height
#                     # predictions[:, :, 2] *= self.img_width
#                     # predictions[:, :, 3] *= self.img_height
#                     # print(predictions)
#                     loss = tf.reduce_mean(tf.square(predictions - y_batch))

#                 gradients = tape.gradient(loss, self.model.trainable_variables)
#                 optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

#                 loss_value = loss.numpy()
#                 loss_history.append(loss_value)
#                 epoch_loss += loss_value

#                 # print(f"Epoch {epoch + 1}/{num_epochs}, Batch {batch + 1}/{data_loader.num_batches}, Loss: {loss_value:.4f}")

#             # Average loss for the epoch
#             avg_epoch_loss = epoch_loss / data_loader.num_batches
#             print(f"Epoch {epoch + 1} Average Train Loss: {avg_epoch_loss:.7f}")
#             self.evaluate(test_dataloader)
            

#         # Plot training loss
#         plt.plot(loss_history, label="Training Loss")
#         plt.xlabel("Batch")
#         plt.ylabel("Loss")
#         plt.title("Training Loss Per Batch")
#         plt.legend()
#         plt.show()
        
#     def evaluate(self, data_loader):
#         """
#         Evaluate the model on the provided DataLoader.

#         Parameters:
#         - data_loader: DataLoader instance, handles loading of evaluation data.
#         """
#         data_loader.reset_batch_pointer()
#         total_loss = 0

#         for batch in tqdm(range(data_loader.num_batches)):
#             x_batch, y_batch = data_loader.next_batch()

#             x_batch = np.array(x_batch).reshape((data_loader.batch_size, self.seq_length, -1))
#             y_batch = np.array(y_batch)
            
#             # x_batch[:,:,0] *= self.img_width
#             # x_batch[:,:,1] *= self.img_height
#             # x_batch[:,:,2] *= self.img_width
#             # x_batch[:,:,3] *= self.img_height
#             y_batch[:,:,0] *= self.img_width
#             y_batch[:,:,1] *= self.img_height
#             y_batch[:,:,2] *= self.img_width
#             y_batch[:,:,3] *= self.img_height

#             predictions = self.model(x_batch).numpy()
#             predictions[:, :, 0] *= self.img_width
#             predictions[:, :, 1] *= self.img_height
#             predictions[:, :, 2] *= self.img_width
#             predictions[:, :, 3] *= self.img_height
#             print(predictions)
#             loss = tf.reduce_mean(tf.square(predictions - y_batch))
#             total_loss += loss.numpy()

#         avg_loss = total_loss / data_loader.num_batches
#         print(f"Average Validation Loss: {avg_loss:.7f}")

#     def predict(self, input_sequence):
#         """
#         Predict the next state given an input sequence.

#         Parameters:
#         - input_sequence: ndarray, input sequence of shape (seq_length, input_dim).

#         Returns:
#         - ndarray, predicted state of shape (output_dim,).
#         """
#         input_sequence = np.expand_dims(input_sequence, axis=0)
#         preds = self.model.predict(input_sequence, verbose=0)[0]
#         preds[:, 0] *= self.img_width
#         preds[:, 1] *= self.img_height
#         preds[:, 2] *= self.img_width
#         preds[:, 3] *= self.img_height
#         return preds
    
#     def multi_predict(self, input_sequence):
#         preds = self.model.predict(input_sequence, verbose=0)
#         preds[:, :, 0] *= self.img_width
#         preds[:, :, 1] *= self.img_height
#         preds[:, :, 2] *= self.img_width
#         preds[:, :, 3] *= self.img_height
#         return preds

#     def save_model(self, file_path):
#         """
#         Save the trained model to a file.

#         Parameters:
#         - file_path: str, path to save the model.
#         """
#         self.model.save(file_path)
        
#     def load_model(self, file_path):
#         """
#         Load a trained model from a file.

#         Parameters:
#         - file_path: str, path to load the model.
#         """
#         self.model = tf.keras.models.load_model(file_path)
#         # print("Model loaded successfully from", file_path)

#     def save_predictions(self, data_loader, file_path):
#         """
#         Save predictions to a file.

#         Parameters:
#         - data_loader: DataLoader instance.
#         - file_path: str, path to save the predictions.
#         """
#         input_data = data_loader.data
#         input_data_padded = tf.keras.preprocessing.sequence.pad_sequences(
#             input_data, padding='post', maxlen=self.seq_length, dtype='float32'
#         )

#         predictions = self.model.predict(input_data_padded)

#         pred_results = {
#             "predictions": predictions,
#             "ground_truth": input_data_padded
#         }

#         with open(file_path, "wb") as f:
#             pickle.dump(pred_results, f)


# if __name__ == "__main__":
#     # Training configuration
#     input_dim = 4
#     output_dim = 4
#     seq_length = 10
#     batch_size = 64
#     num_epochs = 10

#     # Initialize data loader
#     data_loader = DataLoader(batch_size=batch_size, seq_length=seq_length)
#     test_data_loader = DataLoader(batch_size=batch_size, seq_length=seq_length, train=False)

#     # Initialize and train the tracker
#     tracker = LSTMTracker(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length)
#     tracker.load_model("lstm_model.keras")
#     # tracker.train_with_dataloader(data_loader, test_data_loader, num_epochs=num_epochs, batch_size=batch_size)
#     sequence = [[0.1, 0.2, 0.3, 0.4]]
#     sequence = np.pad(sequence, ((0, seq_length - 1), (0, 0)), mode="edge")
#     preds = tracker.predict(sequence)
#     print(preds)

#     # tracker.evaluate(test_data_loader)
#     # Save predictions
#     # tracker.save_predictions(data_loader, "pred_results.pkl")
#     # tracker.save_model("lstm_model.keras")
