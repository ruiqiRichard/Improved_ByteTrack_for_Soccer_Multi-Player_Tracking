import os
import pickle
import numpy as np
import zipfile
import glob
import random

class DataLoader:
    """
    DataLoader for ByteTrack-style multi-object tracking data.
    """

    def __init__(self, batch_size=50, seq_length=5, datasets=[0], forcePreProcess=False):
        """
        Initialize the DataLoader.

        Parameters:
        - batch_size: int, number of samples per batch.
        - seq_length: int, length of each sequence in frames.
        - datasets: list, indices of datasets to use (corresponding to self.data_dirs).
        - forcePreProcess: bool, force reprocessing of data even if preprocessed data exists.
        """
        self.data_dirs = ['./Dataset/tracking']

        try:
            self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        except IndexError as e:
            raise ValueError(f"An index in 'datasets' is out of range: {e}")

        self.data_dir = './Dataset/tracking'
        self.batch_size = batch_size
        self.seq_length = seq_length

        # Ensure that the train.zip is unzipped
        self.extracted_dir = os.path.join(self.data_dir, "temp_extracted")
        zip_path = os.path.join(self.data_dir, 'train.zip')

        if not os.path.exists(self.extracted_dir):
            if os.path.exists(zip_path):
                print(f"Extracting {zip_path}...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.extracted_dir)
            else:
                raise FileNotFoundError(f"{zip_path} not found. Ensure the file exists.")

        data_file = os.path.join(self.data_dir, "trajectories.cpkl")

        if not os.path.exists(data_file) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.preprocess([self.extracted_dir], data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        """
        Preprocess tracking data from ByteTrack-style dataset.
        """
        all_object_data = {}
        dataset_indices = []
        current_object = 0

        for directory in data_dirs:
            for sequence_dir in glob.glob(os.path.join(directory, '*')):
                if not os.path.isdir(sequence_dir):
                    continue

                print(f"Processing sequence: {sequence_dir}")

                for file_name in glob.glob(os.path.join(sequence_dir, '*.txt')):
                    print(f"Reading file: {file_name}")
                    data = np.loadtxt(file_name, delimiter=',')

                    for track_id in np.unique(data[:, 1]):  # Unique track IDs
                        track_data = data[data[:, 1] == track_id]
                        x_center = (track_data[:, 2] + track_data[:, 4]) / 2
                        y_center = (track_data[:, 3] + track_data[:, 5]) / 2
                        frame_ids = track_data[:, 0]
                        traj = np.vstack((frame_ids, x_center, y_center)).T
                        all_object_data[current_object + int(track_id)] = traj

                dataset_indices.append(current_object + len(all_object_data))
                current_object += len(all_object_data)

        complete_data = (all_object_data, dataset_indices)
        with open(data_file, "wb") as f:
            pickle.dump(complete_data, f, protocol=2)

        print(f"Preprocessed data saved to {data_file}")

    def load_preprocessed(self, data_file):
        """
        Load preprocessed data from file.
        """
        with open(data_file, "rb") as f:
            self.raw_data = pickle.load(f)

        all_object_data = self.raw_data[0]
        self.data = []
        counter = 0

        for obj in all_object_data:
            traj = all_object_data[obj]
            if traj.shape[0] > (self.seq_length + 2):
                self.data.append(traj[:, 1:3])
                counter += int(traj.shape[0] / (self.seq_length + 2))

        self.num_batches = int(counter / self.batch_size)

    def next_batch(self):
        """
        Fetch the next batch of sequences.
        """
        x_batch, y_batch = [], []

        for _ in range(self.batch_size):
            traj = self.data[self.pointer]
            n_batch = int(traj.shape[0] / (self.seq_length + 2))
            idx = random.randint(0, traj.shape[0] - self.seq_length - 2)
            x_batch.append(traj[idx:idx + self.seq_length])
            y_batch.append(traj[idx + 1:idx + self.seq_length + 1])

            if random.random() < (1.0 / float(n_batch)):
                self.tick_batch_pointer()

        return x_batch, y_batch

    def tick_batch_pointer(self):
        """
        Advance the pointer to the next data sequence.
        """
        self.pointer += 1
        if self.pointer >= len(self.data):
            self.pointer = 0

    def reset_batch_pointer(self):
        """
        Reset the data pointer to the start.
        """
        self.pointer = 0
