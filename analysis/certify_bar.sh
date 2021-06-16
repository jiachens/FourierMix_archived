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

runtime=250

bank=guests
if [[ "$HOSTNAME" = *"lassen"* ]]; then
  bank=safeml
fi

for cor in 'blue_noise_sample' 'checkerboard_cutout' 'inverse_sparkles' 'lines' 'ripple' 'brownish_noise' 'circular_motion_blur' 'pinch_and_twirl' 'sparkles' 'transverse_chromatic_abberation'; do
for sever in '1' '2' '3' '4' '5'; do
    
bsub -nnodes 1 -G $bank -W $runtime python ../code/certify.py cifar10-c-bar ../models/cifar10/resnet110/noise_0.25/checkpoint.pth.tar 0.25 ../test/cifar10-c-bar/0.25_0.25/${cor}_${sever}.out --path ../data --corruption ${cor} --severity ${sever} --skip 20 --batch 4800 --gpu 0

done
done
