###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2021-06-17 00:37:23
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2021-06-17 01:01:31
### 

e1=$(($2 / 4))
e2=$(($2 / 2))
e3=$(($e1 + $e2))
e4=$2

python code/certify_fourier.py cifar10 models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 test/heatmap/fourier --path ./data --skip 20 --batch 4800 --gpu 0 --start $1 --end $e1  &
python code/certify_fourier.py cifar10 models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 test/heatmap/fourier --path ./data --skip 20 --batch 4800 --gpu 1 --start $(($e1)) --end $e2 &
python code/certify_fourier.py cifar10 models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 test/heatmap/fourier --path ./data --skip 20 --batch 4800 --gpu 2 --start $(($e2)) --end $e3 &
python code/certify_fourier.py cifar10 models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 test/heatmap/fourier --path ./data --skip 20 --batch 4800 --gpu 3 --start $(($e3)) --end $e4 &