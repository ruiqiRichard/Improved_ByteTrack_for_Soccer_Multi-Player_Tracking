# LSTM ByteTrack for Soccer Multi-Player Tracking

This project adapts ByteTrack for soccer player tracking using the SoccerNet dataset. It explores alternative approaches for improving multi-object tracking accuracy in complex scenarios like soccer games, but the results indicate that not all modifications yield improvements.

## Purpose

The primary aim is to investigate and analyze enhancements to ByteTrack to address challenges such as:

* Non-linear motion modeling.
* Frequent occlusions in dynamic environments.
* Dense overlaps among tracked objects.

Proposed modifications include:

* Replacing the Kalman filter with an LSTM model for improved motion prediction.
* Using Distance IOU (DIoU) for more precise overlap measurements.
* Introducing a decay weight for unmatched track management.

## Prerequisites

Ensure that CUDA and CUDNN are installed and properly configured. Verified environment:

* **OS:** Ubuntu 20.04.3 LTS
* **Python Environment:** Anaconda 3-5.1.0
* **CUDA:** Version 10.0.130
* **CUDNN:** Version 7.6.5

Install dependencies:

```bash
pip install -r requirements.txt
```

## Training LSTM with TensorFlow

Train the LSTM model with the following command:

```bash
cd LSTM
python model_tf.py --data_file <ByteTrack_HOME>/Dataset/tracking --output_model ./lstm_model.keras
```

## MOT20 Test Model

Train on CrowdHuman and MOT20 datasets, and evaluate on MOT20 train. Pretrained models:

| Model | MOTA | IDF1 | IDs | FPS |
|-------|------|------|-----|-----|
| bytetrack_x_mot20 [baidu (code:3apd)] | 93.4 | 89.3 | 1057 | 17.5 |

Place the pretrained model in the `/pretrained` directory.

## Running Inference

Run inference for each sequence:

1. Set up the environment:

```bash
export ByteTrack_HOME=<ByteTrack_HOME>
cd <ByteTrack_HOME>
export SN_TRACKING_MODE=test
bash run_bytetrack_gt_batch.sh
```

2. For challenge mode:

```bash
export SN_TRACKING_MODE=challenge
```

## Local Evaluation

To evaluate results locally:

1. **Generate `gt.zip`:**

```bash
python eval/zip_gt.py -f Dataset/tracking/test/
```

2. **Zip Tracker Results:**

```bash
cd <RESULT_FOLDER> # e.g., YOLOX_outputs/yolox_x_soccernet_no_gt/track_vis
zip soccernet_mot_results.zip SNMOT-???.txt
```

3. **Prepare for Evaluation:** Move `gt.zip` and `soccernet_mot_results.zip` to the `eval` folder:

```bash
cd eval
```

4. **Run Evaluation:** Install the evaluation tool and execute the script:

```bash
pip install git+https://github.com/JonathonLuiten/TrackEval.git

python evaluate_soccernet_v3_tracking.py \
  --TRACKERS_FOLDER_ZIP soccernet_mot_results.zip \
  --GT_FOLDER_ZIP gt.zip \
  --BENCHMARK SNMOT \
  --DO_PREPROC False \
  --SEQMAP_FILE SNMOT-test.txt \
  --TRACKERS_TO_EVAL test \
  --SPLIT_TO_EVAL test \
  --OUTPUT_SUB_FOLDER eval_results
```

## Key Features

* **Motion Modeling:** Integration of LSTM to explore non-linear motion tracking.
* **Overlap Metrics:** Use of Distance IOU to handle dense overlaps.
* **Track Retention:** Implementation of a decay weight mechanism for unmatched tracks.
* **Comprehensive Evaluation:** Supports metrics such as HOTA, MOTA, and ASSA.
