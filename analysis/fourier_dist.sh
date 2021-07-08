###
 # @Description: 
 # @Autor: Jiachen Sun
 # @Date: 2021-07-02 16:53:28
 # @LastEditors: Jiachen Sun
 # @LastEditTime: 2021-07-02 16:53:52
### 
for severity in 1 2 3 4 5; do
for cor in 'fog' 'contrast' 'gaussian_blur' 'gaussian_noise' 'shot_noise' 'impulse_noise' 'defocus_blur' 'frosted_glass_blur' 'motion_blur' 'zoom_blur' 'snow' 'frost' 'brightness' 'elastic' 'pixelate' 'jpeg_compression';
# for cor in 'fog' 'contrast';
do
    python ./code/fourier_dist.py cifar10-c --path ./data --corruption ${cor} --severity ${severity} --gpu 1
done
done

for severity in 1 2 3 4 5; do
for cor in 'blue_noise_sample' 'brownish_noise' 'checkerboard_cutout' 'circular_motion_blur' 'inverse_sparkles' 'lines' 'pinch_and_twirl' 'ripple' 'sparkles' ' transverse_chromatic_abberation';
do
    python ./code/fourier_dist.py cifar10-c-bar --path ./data --corruption ${cor} --severity ${severity} --gpu 1
done
done
