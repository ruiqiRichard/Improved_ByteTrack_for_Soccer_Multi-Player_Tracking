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

## Install ByteTrack

### 1. Installing on the host machine
Step1. Install ByteTrack.
```shell
git clone https://github.com/ifzhang/ByteTrack.git
cd ByteTrack
pip3 install -r requirements.txt
python3 setup.py develop
```

Step2. Install [pycocotools](https://github.com/cocodataset/cocoapi).

```shell
pip3 install cython; pip3 install 'git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI'
```

Step3. Others
```shell
pip3 install cython_bbox
```
### 2. Docker build
```shell
docker build -t bytetrack:latest .

# Startup sample
mkdir -p pretrained && \
mkdir -p YOLOX_outputs && \
xhost +local: && \
docker run --gpus all -it --rm \
-v $PWD/pretrained:/workspace/ByteTrack/pretrained \
-v $PWD/datasets:/workspace/ByteTrack/datasets \
-v $PWD/YOLOX_outputs:/workspace/ByteTrack/YOLOX_outputs \
-v /tmp/.X11-unix/:/tmp/.X11-unix:rw \
--device /dev/video0:/dev/video0:mwr \
--net=host \
-e XDG_RUNTIME_DIR=$XDG_RUNTIME_DIR \
-e DISPLAY=$DISPLAY \
--privileged \
bytetrack:latest
```

## Install dependencies:

```bash
pip install -r requirements.txt
```

## Training LSTM with PyTorch

Train the LSTM model with the following command:

```bash
cd LSTM
python model.py
```
model will be saved at ./LSTM

## MOT20 Test Model

Train on CrowdHuman and MOT20 datasets, and evaluate on MOT20 train. Pretrained models:


| Model    |  MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|bytetrack_x_mot20 [[google]](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing), [[baidu(code:3apd)]](https://pan.baidu.com/s/1bowJJj0bAnbhEQ3_6_Am0A) | 93.4 | 89.3 | 1057 | 17.5 |


Place the pretrained model in the `/pretrained` directory.

## Running Inference

Run inference for each sequence:

1. Set up the environment:

```bash
export ByteTrack_HOME=<ByteTrack_HOME>
cd <ByteTrack_HOME>
export SN_TRACKING_MODE=test
bash run_bytetrack_no_gt_batch.sh
```
For detailed arguments, see tools/demo_track_no_gt.py
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
