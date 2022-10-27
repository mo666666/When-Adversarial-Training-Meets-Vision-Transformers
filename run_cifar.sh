CUDA_VISIBLE_DEVICES=0 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_vanilla" --method 'pgd' --seed 0  &
CUDA_VISIBLE_DEVICES=1 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./pgd_architecture" --method 'pgd' --seed 0 --ARD --PRM &
CUDA_VISIBLE_DEVICES=2 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_vanilla" --method 'trades' --seed 0  &
CUDA_VISIBLE_DEVICES=3 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./trades_architecture" --method 'trades' --seed 0 --ARD --PRM &
CUDA_VISIBLE_DEVICES=4 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_vanilla" --method 'MART' --seed 0  &
CUDA_VISIBLE_DEVICES=5 python train_cifar.py --model "deit_tiny_patch16_224" --n_w 10 --out-dir "./mart_architecture" --method 'MART' --seed 0 --ARD --PRM &
wait