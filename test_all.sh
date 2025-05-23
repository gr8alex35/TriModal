# Evaluation-only

CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_start --use_end --use_image --eval_only      --log_path logs/data_all_eval_StartEndImage.log  > logs/data_all_eval_StartEndImage.log 2>&1 &

CUDA_VISIBLE_DEVICES=0 nohup python main.py --use_start --use_end --eval_only                  --log_path logs/data_all_eval_StartEnd.log       > logs/data_all_eval_StartEnd.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_start --use_image --eval_only                --log_path logs/data_all_eval_StartImage.log     > logs/data_all_eval_StartImage.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_end --use_image --eval_only                  --log_path logs/data_all_eval_EndImage.log       > logs/data_all_eval_EndImage.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_start --eval_only                            --log_path logs/data_all_eval_StartOnly.log      > logs/data_all_eval_StartOnly.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_end --eval_only                              --log_path logs/data_all_eval_EndOnly.log        > logs/data_all_eval_EndOnly.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 nohup python main.py --use_image --eval_only                            --log_path logs/data_all_eval_ImageOnly.log      > logs/data_all_eval_ImageOnly.log 2>&1 &
