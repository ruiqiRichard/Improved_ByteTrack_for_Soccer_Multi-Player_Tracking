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

    def __init__(self, batch_size=50, seq_length=10, datasets=[0], forcePreProcess=False, train = True):
        """
        Initialize the DataLoader.

        Parameters:
        - batch_size: int, number of samples per batch.
        - seq_length: int, length of each sequence in frames.
        - datasets: list, indices of datasets to use (corresponding to self.data_dirs).
        - forcePreProcess: bool, force reprocessing of data even if preprocessed data exists.
        """
        self.data_dirs = ['../Dataset/tracking']
        self.flow_data_dirs = ['../Model/optical_flow_features'] if train else ['../Model/optical_flow_features_test']

        try:
            self.used_data_dirs = [self.data_dirs[x] for x in datasets]
        except IndexError as e:
            raise ValueError(f"An index in 'datasets' is out of range: {e}")

        self.data_dir = '../Dataset/tracking'
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.width = 1920
        self.height = 1080
        
        self.data = {'traj':[], 'flow':[]}

        # Ensure that the train.zip is unzipped
        # self.extracted_dir = os.path.join(self.data_dir, "temp_extracted")
        # zip_path = os.path.join(self.data_dir, 'train.zip')

        # if not os.path.exists(self.extracted_dir):
        #     if os.path.exists(zip_path):
        #         print(f"Extracting {zip_path}...")
        #         with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        #             zip_ref.extractall(self.extracted_dir)
        #     else:
        #         raise FileNotFoundError(f"{zip_path} not found. Ensure the file exists.")

        data_file = os.path.join(self.data_dir, "trajectories_train.cpkl") if train else os.path.join(self.data_dir, "trajectories_test.cpkl")

        if not os.path.exists(data_file) or forcePreProcess:
            print("Creating pre-processed data from raw data")
            self.preprocess([os.path.join(self.data_dir, "train") if train else os.path.join(self.data_dir, "test")], data_file)

        self.load_preprocessed(data_file)
        self.reset_batch_pointer()

    def preprocess(self, data_dirs, data_file):
        """
        Preprocess tracking data from ByteTrack-style dataset, specifically focusing on .txt files in /gt directories.
        """
        all_object_data = {}
        dataset_indices = []
        current_object = 0

        for directory in data_dirs:
            print(f"Processing directory: {directory}")

            # Recursively traverse the directories
            for root, dirs, files in os.walk(directory):
                # Only process files in 'gt' directories
                
                if 'gt' in root:
                    print(root)
                    print(f"Processing directory: {root}")
                    for file_name in files:
                        if file_name.endswith('.txt'):
                            
                            file_path = os.path.join(root, file_name)
                            dir_name = root.split("train")[-1][1:-3] if "train" in root else root.split("test")[-1][1:-3]
                            # print(dir_name)
                            print(f"Reading file: {file_path}")
                            
                            # Load data from the .txt file
                            data = np.loadtxt(file_path, delimiter=',')

                            # Process tracking data
                            for track_id in np.unique(data[:, 1]):  # Unique track IDs
                                track_data = data[data[:, 1] == track_id]
                                tl_x = track_data[:, 2] / self.width
                                tl_y = track_data[:, 3] / self.height
                                width = track_data[:, 4] / self.width
                                height = track_data[:, 5] / self.height
                                # print(tl_x)
                                frame_ids = track_data[:, 0]
                                # print(dirs)
                                # directories = [os.path.basename(os.path.normpath(directory))] * len(frame_ids)
                                # print(directories)
                                traj = np.vstack(([dir_name] * len(frame_ids), frame_ids, tl_x, tl_y, width, height)).T
                                all_object_data[current_object + int(track_id)] = traj

                            print(f"Processed {len(np.unique(data[:, 1]))} tracks from file {file_name}")

                    # Add dataset index after processing each directory
                    dataset_indices.append(current_object + len(all_object_data))
                    current_object += len(all_object_data)

        # Save the processed data to a file
        complete_data = (all_object_data, dataset_indices)
        # print(complete_data)
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
        # print(len(all_object_data))

        counter = 0

        for obj in all_object_data:
            traj = all_object_data[obj]
            # print(traj.shape)
            if traj.shape[0] >= (self.seq_length + 1):
                self.data['traj'].append(traj[:, 2:])
                frame_ids = traj[:, 1].astype(float)
                # print(frame_ids.shape)
                flow = np.load(os.path.join(self.flow_data_dirs[0], traj[0, 0] + "_optical_flow.npy"))

                zero_row = np.zeros((1, flow.shape[1]))
                flow = np.vstack((zero_row, flow))
                # print(flow.shape)
                flow = flow[frame_ids.astype(int)-1]
                
                
                self.data['flow'].append(flow)
                # print("loading flow", os.path.join(self.flow_data_dirs[0], traj[0, 0] + "_optical_flow.npy"))
                counter += int(traj.shape[0] / (self.seq_length + 1))

        self.num_batches = int(counter / self.batch_size)
        # print(self.num_batches)

    def next_batch(self):
        """
        Fetch the next batch of sequences.
        """
        x_batch, y_batch = {'traj':[], 'flow':[]}, []

        for _ in range(self.batch_size):
            traj = self.data['traj'][self.pointer].astype(float)
            # print(traj.shape)
            n_batch = int(traj.shape[0] / (self.seq_length + 1))
            idx = random.randint(0, traj.shape[0] - self.seq_length - 1)
            x_batch['traj'].append(traj[idx:idx + self.seq_length])
            x_batch['flow'].append(self.data['flow'][self.pointer][idx:idx + self.seq_length])
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

if __name__ == "__main__":
    # Training configuration
    input_dim = 4
    output_dim = 4
    seq_length = 20
    batch_size = 64
    num_epochs = 10

    # Initialize data loader
    # train_loader = DataLoader(batch_size=batch_size, seq_length=seq_length)
    test_loader = DataLoader(batch_size=batch_size, seq_length=seq_length, train=False)
    
    print(test_loader.next_batch()[1][0].shape)