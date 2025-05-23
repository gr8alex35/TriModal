# Train and Evaluation

CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_start --use_end --use_image     --log_path logs/data_all_StartEndImage.log     > logs/data_all_StartEndImage.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_start --use_end                 --log_path logs/data_all_StartEnd.log         > logs/data_all_StartEnd.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_start --use_image               --log_path logs/data_all_StartImage.log       > logs/data_all_StartImage.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_end --use_image                 --log_path logs/data_all_EndImage.log         > logs/data_all_EndImage.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_start                          --log_path logs/data_all_StartOnly.log        > logs/data_all_StartOnly.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_end                            --log_path logs/data_all_EndOnly.log          > logs/data_all_EndOnly.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_image                          --log_path logs/data_all_ImageOnly.log        > logs/data_all_ImageOnly.log 2>&1 &
