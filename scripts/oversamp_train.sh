python main.py\
        --base_dir="/repo/course/sem21_01/PCB-fault-classification/"\
        --label_dir="/repo/course/sem21_01/PCB-fault-classification/label.csv" \
        --ckpt_path="/repo/course/sem21_01/PCB-fault-classification/ckpt/"\
        --mode="train"\
        --lr_type='multi'\
        --model_index=5 \
        --verbose=20 \
        --epochs=30
        
## model_index = 5 >>> mode="train", lr_type='multi', epochs=30