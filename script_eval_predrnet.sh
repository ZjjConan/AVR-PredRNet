
python main.py --dataset-name I-RAVEN --dataset-dir your_dataset_root_dir --gpu 0,1,2,3 \
               --image-size 80 -a predrnet_raven --num-extra-stages 3 \
               -e --resume your_checkpoint_dir/model_best.pth.tar \
               --show-detail