
# ------------------------------------ PredRNet ------------------------------------------------

# for supervised PredRNet on RAVEN, and Clevr-Matrix
python main.py --dataset-name "dataset_name" --dataset-dir "your_dataset_root_dir" --fp16 \
    --gpu 0,1,2,3 --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 \
    --wd 1e-5 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
    --ckpt ckpts/ --print-freq 100

# for supervised PredRNet on VADs
python main.py --dataset-name "dataset_name" --dataset-dir "your_dataset_root_dir" --fp16 \
    --gpu 0,1,2,3 --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 \
    --wd 1e-5 -a predrnet_vad --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
    --ckpt ckpts/ --print-freq 100

# for supervised PredRNet on all PGMs
# We recommend setting the weight decay to 1e-7 or 1e-8. This was found to be optimal for stabilizing the training of the model.
python main.py --dataset-name "dataset_name" --dataset-dir "your_dataset_root_dir" --fp16 \
    --gpu 0,1,2,3 --image-size 80 --epochs 100 --seed 12345 --batch-size 256 --lr 0.001 \
    --wd 1e-7 -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.5 \
    --ckpt ckpts/ --print-freq 200 --enable-rc


# ----------------------------------------------------------------------------------------------
# ---------------------------------- SSPredRNet ------------------------------------------------

# for self-supervised SSPredRNet on RAVEN, and CLEVR-Matrix
python main_ssl.py --dataset-name "dataset_name" --dataset-dir "your_dataset_root_dir" --fp16 \
    --gpu 0,1,2,3 --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 \
    --wd 1e-5 -a sspredrnet_raven --num-extra-stages 3 --block-drop 0.1 \
    --ckpt ckpts/ --print-freq 100 --c-margin 0.7


# for self-supervised SSPredRNet on VADs
python main_ssl.py --dataset-name "dataset_name" --dataset-dir "your_dataset_root_dir" --fp16 \
    --gpu 0,1,2,3 --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 \
    --wd 1e-5 -a sspredrnet_vad --num-extra-stages 3 --block-drop 0.1 \
    --ckpt ckpts/ --print-freq 100 --c-margin 0.7


# for self-supervised PredRNet on all PGMs
python main_ssl.py --dataset-name "dataset_name" --dataset-dir "your_dataset_root_dir" --fp16 \
    --gpu 0,1,2,3 --image-size 80 --epochs 100 --seed 12345 --batch-size 256 --lr 0.001 \
    --wd 1e-5 -a sspredrnet_raven --num-extra-stages 3 --block-drop 0.1 \
    --ckpt ckpts/ --print-freq 200 --c-margin 0.7 --enable-rc