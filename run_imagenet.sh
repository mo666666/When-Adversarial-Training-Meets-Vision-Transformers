python train_imagenet.py --model "vit_base_patch16_224_in21k" --out-dir "./pgd_vanilla" --seed 0
python train_imagenet.py --model "vit_base_patch16_224_in21k" --n_w 2 --out-dir "./pgd_architecture" --seed 0 --ARD --PRM
python train_imagenet.py --model "swin_base_patch4_window7_224_in22k" --out-dir "./pgd_vanilla" --seed 0
python train_imagenet.py --model "swin_base_patch4_window7_224_in22k" --n_w 2 --out-dir "./pgd_architecture" --seed 0 --ARD --PRM