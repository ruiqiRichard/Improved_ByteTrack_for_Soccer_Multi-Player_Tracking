import os
import torch
import torch.nn as nn
import numpy as np

class CrossAttentionModel(nn.Module):
    """
    A model that combines optical flow and LSTM embeddings using cross-attention.
    """

    def __init__(self, embedding_dim, flow_dim, attention_dim):
        super(CrossAttentionModel, self).__init__()
        self.query_projection = nn.Linear(embedding_dim, attention_dim)
        self.key_projection = nn.Linear(flow_dim, attention_dim)
        self.value_projection = nn.Linear(flow_dim, attention_dim)
        self.output_projection = nn.Linear(attention_dim, embedding_dim)

    def forward(self, lstm_embeddings, optical_flow_features):
        queries = self.query_projection(lstm_embeddings)
        keys = self.key_projection(optical_flow_features)
        values = self.value_projection(optical_flow_features)

        attention_scores = torch.bmm(queries, keys.transpose(1, 2))
        attention_weights = torch.softmax(attention_scores, dim=-1)
        context = torch.bmm(attention_weights, values)

        combined_embeddings = self.output_projection(context)
        return combined_embeddings


class LSTMTracker(nn.Module):
    """
    An LSTM-based model for predicting object trajectories.
    """

    def __init__(self, input_dim=4, output_dim=4, seq_length=10, lstm_units=64):
        super(LSTMTracker, self).__init__()
        self.seq_length = seq_length
        self.lstm = nn.LSTM(input_dim, lstm_units, batch_first=True)
        self.fc = nn.Linear(lstm_units, output_dim)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.fc(lstm_out)


if __name__ == "__main__":
    # Directory containing optical flow features
    optical_flow_dir = "./Model/optical_flow_features"

    # Load all .npy files in the directory into a single tensor
    optical_flow_features_list = [
        np.load(os.path.join(optical_flow_dir, file))
        for file in os.listdir(optical_flow_dir) if file.endswith(".npy")
    ]
    optical_flow_features = torch.tensor(np.concatenate(optical_flow_features_list), dtype=torch.float32)

    input_dim = 4  # Matches optical flow feature dimensions
    output_dim = 4
    seq_length = 10
    lstm_units = 64
    lstm_model = LSTMTracker(input_dim=input_dim, output_dim=output_dim, seq_length=seq_length, lstm_units=lstm_units)

    # Load the state dictionary
    lstm_model_path = "./LSTM/lstm_model.pth"
    state_dict = torch.load(lstm_model_path, map_location=torch.device('cpu'))
    lstm_model.load_state_dict(state_dict)
    lstm_model.eval()  # Set to evaluation mode

    # Generate LSTM embeddings
    batch_size, seq_length, flow_dim = optical_flow_features.size(0), seq_length, input_dim
    optical_flow_features = optical_flow_features.view(batch_size, seq_length, flow_dim)  # Reshape for LSTM input
    lstm_embeddings = lstm_model(optical_flow_features)

    # Initialize and test the CrossAttentionModel
    embedding_dim = lstm_embeddings.size(-1)
    cross_attention = CrossAttentionModel(
        embedding_dim=embedding_dim,
        flow_dim=optical_flow_features.size(-1),
        attention_dim=32
    )
    combined_embeddings = cross_attention(lstm_embeddings, optical_flow_features)

    print("Combined Embeddings Shape:", combined_embeddings.shape)
