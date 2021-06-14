###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2021-06-14 17:31:15
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2021-06-14 17:42:51
### 
for cor in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'frosted_glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'brightness' 'elastic' 'pixelate' 'jpeg_compression';
do
    python ../code/certify.py cifar10-c ../models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 ../test/prelin/noise_0.25_0.25/${cor}_5.out --path ../data --corruption ${cor} --severity 5 --skip 20 --batch 2400 --gpu 2
done