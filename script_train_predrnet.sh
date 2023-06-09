
python main.py --dataset-name I-RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 --fp16 \
               --image-size 80 --epochs 100 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-5 \
               -a predrnet_raven --num-extra-stages 3 --block-drop 0.1 --classifier-drop 0.1 \
               --ckpt ckpts/

# python main.py --dataset-name PGM/interpolation --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 --fp16 \
#                --image-size 80 --num-classes 1 --epochs 100 -p 200 --seed 12345 -a predrnet_raven --num-extra-stages 3 \
#                --classifier-hidreduce 4 --block-drop 0.1 --classifier-drop 0.1 \
#                --batch-size 256 --lr 0.001 --wd 1e-7 \
#                --ckpt ckpts/

# python main.py --dataset-name Analogy/interpolation --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 \
#                --image-size 80 --epochs 3 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-7 -p 200 \
#                -a predrnet_analogy --num-extra-stages 3 --block-drop 0.0 --classifier-drop 0.1 \
#                --ckpt ckpts/

# python main.py --dataset-name CLEVR-Matrix --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 --fp16 \
#                --image-size 80 --epochs 200 --seed 12345 --batch-size 128 --lr 0.001 --wd 1e-8 \
#                -a predrnet_raven --num-extra-stages 3 --block-drop 0.0 --classifier-drop 0.1 \
#                --ckpt ckpts/ --in-channels 3