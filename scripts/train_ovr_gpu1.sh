python main.py\
        --base_dir="/repo/course/sem21_01/PCB-fault-classification/"\
        --label_dir="/repo/course/sem21_01/PCB-fault-classification/label.csv" \
        --ckpt_path="/repo/course/sem21_01/PCB-fault-classification/ckpt/"\
        --mode="test"\
        --lr_type='multi'\
        --model_index=55 \
        --verbose=20 \
        --epochs=50 \
        --device_index=1
        
## model_index = 5 >>> mode="train", lr_type='multi', epochs=50, base_model=plain_resnet50
## model_index = 55 >>> mode="train_ovr", lr_type='multi', epochs=50, base_model=plain_resnet50