import cv2
import numpy as np
import os
import glob
import matplotlib.pyplot as plt

class OpticalFlowExtractor:
    def __init__(self, max_corners=100, quality_level=0.3, min_distance=7, block_size=7, flow_dim=200):
        """
        Initialize the OpticalFlowExtractor.

        Parameters:
        - max_corners: int, maximum number of corners to detect.
        - quality_level: float, quality level for corner detection.
        - min_distance: float, minimum distance between detected corners.
        - block_size: int, block size for corner detection.
        - flow_dim: int, fixed dimension for optical flow vectors (for padding/truncation).
        """
        self.feature_params = dict(
            maxCorners=max_corners,
            qualityLevel=quality_level,
            minDistance=min_distance,
            blockSize=block_size,
        )
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03),
        )
        self.flow_dim = flow_dim  # Fixed dimension for optical flow vectors

    def extract_flow_from_frames(self, frames):
        """
        Extract optical flow features from a sequence of frames.

        Parameters:
        - frames: list of np.ndarray, sequence of image frames.

        Returns:
        - np.ndarray, optical flow features for the given frames.
        """
        if len(frames) < 2:
            raise ValueError("Not enough frames to calculate optical flow.")
        
        prev_gray = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
        p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **self.feature_params)
        
        optical_flows = []
        for i in range(1, len(frames)):
            next_gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY)
            p1, st, err = cv2.calcOpticalFlowPyrLK(prev_gray, next_gray, p0, None, **self.lk_params)
            good_old = p0[st == 1]
            good_new = p1[st == 1]
            flow = (good_new - good_old).flatten()

            # Pad or truncate the flow to the fixed dimension
            if len(flow) < self.flow_dim:
                flow = np.pad(flow, (0, self.flow_dim - len(flow)), mode='constant')
            else:
                flow = flow[:self.flow_dim]

            optical_flows.append(flow)
            prev_gray = next_gray.copy()
            p0 = good_new.reshape(-1, 1, 2)
        
        return np.array(optical_flows)

    def process_dataset(self, dataset_dir, output_dir, num_frames=31):
        """
        Process each subdirectory in the dataset to extract optical flow features.

        Parameters:
        - dataset_dir: str, path to the dataset directory.
        - output_dir: str, path to save the extracted optical flow features.
        - num_frames: int, number of frames to process per sequence.
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        for sub_dir in os.listdir(dataset_dir):
            sub_dir_path = os.path.join(dataset_dir, sub_dir)
            if not os.path.isdir(sub_dir_path):
                continue

            img1_dir = os.path.join(sub_dir_path, "img1")
            if not os.path.exists(img1_dir):
                print(f"Skipping {sub_dir} (no 'img1' directory found)")
                continue

            frame_files = sorted(glob.glob(os.path.join(img1_dir, "*.jpg")))[:num_frames]
            frames = [cv2.imread(frame_file) for frame_file in frame_files if os.path.exists(frame_file)]
            
            if len(frames) < 2:
                print(f"Not enough frames in {img1_dir} to compute optical flow.")
                continue

            # Extract optical flow for the frames
            optical_flows = self.extract_flow_from_frames(frames)

            # Save optical flow features as a .npy file
            output_file = os.path.join(output_dir, f"{sub_dir}_optical_flow.npy")
            np.save(output_file, optical_flows)
            print(f"Saved optical flow for {sub_dir} to {output_file}.")

    def visualize_optical_flow(self, flow_file, save_dir=None):
        """
        Visualize optical flow features stored in a .npy file.

        Parameters:
        - flow_file: str, path to the .npy file containing optical flow features.
        - save_dir: str or None, directory to save visualizations or None to just display them.
        """
        flow_data = np.load(flow_file)
        num_frames, flow_dim = flow_data.shape
        num_points = flow_dim // 2  # Assume 2D flow (x, y)

        # Reshape the flow data
        flow_vectors = flow_data.reshape(num_frames, num_points, 2)

        for frame_idx, flow in enumerate(flow_vectors):
            # Create a grid for quiver plot
            grid_size = int(np.sqrt(num_points))
            if grid_size**2 != num_points:
                print(f"Skipping visualization for frame {frame_idx}: incompatible flow dimensions.")
                continue

            x = np.linspace(0, 1, grid_size)
            y = np.linspace(0, 1, grid_size)
            xv, yv = np.meshgrid(x, y)

            u = flow[:, 0].reshape(grid_size, grid_size)
            v = flow[:, 1].reshape(grid_size, grid_size)

            plt.figure(figsize=(6, 6))
            plt.quiver(xv, yv, u, v, angles="xy", scale_units="xy", scale=1, color="blue")
            plt.title(f"Optical Flow - Frame {frame_idx + 1}")
            plt.xlabel("X")
            plt.ylabel("Y")
            plt.xlim(0, 1)
            plt.ylim(0, 1)

            if save_dir:
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                output_file = os.path.join(save_dir, f"frame_{frame_idx + 1}.png")
                plt.savefig(output_file)
                print(f"Saved visualization: {output_file}")
            else:
                plt.show()

            plt.close()

if __name__ == "__main__":
    dataset_dir = "./Dataset/tracking/temp_extracted/train"
    output_dir = "./Model/optical_flow_features"
    visualization_dir = "./Model/optical_flow_visualizations"

    extractor = OpticalFlowExtractor()
    extractor.process_dataset(dataset_dir, output_dir)

    # Visualize all optical flow features
    for file in os.listdir(output_dir):
        if file.endswith(".npy"):
            flow_file_path = os.path.join(output_dir, file)
            print(f"Visualizing: {file}")
            extractor.visualize_optical_flow(flow_file_path, save_dir=visualization_dir)
