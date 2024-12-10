ByteTrack_HOME="."
SN_TRACKING_MODE="test"
for file in $ByteTrack_HOME/Dataset/tracking/$SN_TRACKING_MODE/*/ ; do 
    file="${file%/}" # strip trailing slash
    file="${file##*/}"
    # echo "$file is the directoryname without slashes"
    python tools/demo_track.py image -f exps/example/mot/yolox_x_soccernet.py \
    -c pretrained/bytetrack_x_mot20.tar --fp16 --fuse --save_result \
    --device gpu \
    # --lstm_weights $ByteTrack_HOME/LSTM/lstm_model.pth \
    --path $ByteTrack_HOME/Dataset/tracking/$SN_TRACKING_MODE/$file/img1
done