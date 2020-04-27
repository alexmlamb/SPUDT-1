#!/bin/sh

python3 src/main.py --exp-name vmt3 --cuda --run-id mnist-svhn vmt --dataset1 mnist --dataset2 svhn --dw 0.01 --svw 1 --tvw 0.06 --tcw 0.06 --smw 1 --tmw 0.06 --h-dim 256


