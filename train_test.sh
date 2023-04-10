CUDA_VISIBLE_DEVICES=0 python3 train_and_test.py --gpu --img_model resnet50 --save_model --save_dir models/bert_res_hard.pt

CUDA_VISIBLE_DEVICES=0 python3 train_and_test.py --gpu --text_model roberta-base --save_model --save_dir models/roberta_vitb_hard.pt