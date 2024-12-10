# Improved ByteTrack for Soccer Multi-Player Tracking
This is an improved version of ByteTrack for player tracking in soccer games, built upon SoocerNettracking dataset.

### Prerequisite
Make sure CUDA and CUDNN are properly installed and related environment variables are properly set. 
Verified versions are anaconda3-5.1.0 + CUDA 10.0.130 + CUDNN 7.6.5 on Ubuntu 20.04.3 LTS. 

### Training LSTM using Tensorflow
```
    cd LSTM
    python model_tf.py --data_file <ByteTrack_HOME>/Dataset/tracking --output_model ./lstm_model.keras
```

### MOT20 test model

Train on CrowdHuman and MOT20, evaluate on MOT20 train. Download this pretrained model and put it under /pretrained.


| Model    |  MOTA | IDF1 | IDs | FPS |
|------------|-------|------|------|------|
|bytetrack_x_mot20 [[google]](https://drive.google.com/file/d/1HX2_JpMOjOIj1Z9rJjoet9XNy_cCAs5U/view?usp=sharing), [[baidu(code:3apd)]](https://pan.baidu.com/s/1bowJJj0bAnbhEQ3_6_Am0A) | 93.4 | 89.3 | 1057 | 17.5 |

### Run inference for each sequence
```
    export ByteTrack_HOME=<ByteTrack_HOME>
    cd <ByteTrack_HOME>
    export SN_TRACKING_MODE=test
    bash run_bytetrack_gt_batch.sh
```
To run challenge you should set the environment variable differently:
```
    export SN_TRACKING_MODE=challenge
```

### Evaluate locally

Generate gt.zip needed for evaluation
```
    python eval/zip_gt.py -f Dataset/tracking/test/

```

Zip  the Tracker Result
```
    cd <RESULT_FOLDER> # For me is YOLOX_outputs/yolox_x_soccernet_no_gt/track_vis
    zip soccernet_mot_results.zip SNMOT-???.txt
```

Before you run evaluation, move both of your zip file to eval folder

Run evaluation.

```
pip install git+https://github.com/JonathonLuiten/TrackEval.git

python evaluate_soccernet_v3_tracking.py     
--TRACKERS_FOLDER_ZIP soccernet_mot_results.zip     
--GT_FOLDER_ZIP gt.zip     
--BENCHMARK SNMOT     
--DO_PREPROC False     
--SEQMAP_FILE SNMOT-test.txt     
--TRACKERS_TO_EVAL test     
--SPLIT_TO_EVAL test     
--OUTPUT_SUB_FOLDER eval_results
```
