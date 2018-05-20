mpirun -np 2 \
    -H localhost:2 \
    -bind-to none -map-by slot \
    -x NCCL_DEBUG=INFO -x LD_LIBRARY_PATH -x PATH \
    -mca pml ob1 -mca btl ^openib \
    python run_imagenet_train_horovod.py \
    --data_root /mnt/local/data \
    --model resnet-50-v1-b128
