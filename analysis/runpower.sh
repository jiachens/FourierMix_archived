#!/bin/sh 
##usage: ./runall.sh image_size batch_size
##example: ./runall.sh 256 32

runtime=720

bank=guests
if [[ "$HOSTNAME" = *"lassen"* ]]; then
  bank=safeml
fi

bsub -nnodes 1 -G $bank -W $runtime 
