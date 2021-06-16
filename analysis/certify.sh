###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2021-06-14 17:31:15
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2021-06-14 17:42:51
### 

#!/bin/sh
##usage: ./runall.sh image_size batch_size
##example: ./runall.sh 256 32

runtime=400

bank=guests
if [[ "$HOSTNAME" = *"lassen"* ]]; then
  bank=safeml
fi

#for cor in 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'frosted_glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'brightness' 'elastic' 'pixelate' 'jpeg_compression'; do
for cor in 'fog' 'contrast'; do
for sever in '1' '2' '3' '4' '5'; do
    
bsub -nnodes 1 -G $bank -W $runtime python ../code/certify.py cifar10-c ../models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 ../test/cifar10-c/0.25_0.25/${cor}_${sever}.out --path ../data --corruption ${cor} --severity ${sever} --skip 20 --batch 4800 --gpu 0

done
done
