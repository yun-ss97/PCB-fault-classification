python main.py\
        --base_dir="/repo/course/sem21_01/PCB-fault-classification/"\
        --label_dir="/repo/course/sem21_01/PCB-fault-classification/label.csv" \
        --ckpt_path="/repo/course/sem21_01/PCB-fault-classification/ckpt/"\
        --mode="train"\
        --lr_type='multi'\
        --base_model='plain_efficientnetb5'\
        --model_index=7 \
        --verbose=20 \
        --epochs=50 \
        --device_index=1\
        --batch_size=2
        
##     parser.add_argument("--base_model", type=str, default="plain_resnet50", help="[plain_resnet50, custom_resnet50, plain_efficientnetb4, plain_efficientnetb5, plain_efficientnetb7]")

## model_index = 5 >>> mode="train", lr_type='multi', epochs=50, base_model=plain_resnet50, --batch_size=8 (default)
## model_index = 6 >>> mode="train", lr_type='multi', epochs=50, base_model=plain_efficientnetb4, --batch_size=8 (default)
## model_index = 7 >>> >>> mode="train", lr_type='multi', epochs=50, base_model=plain_efficientnetb5, --batch_size=2

## model_index = 55 >>> mode="train_ovr", lr_type='multi', epochs=50, base_model=plain_resnet50