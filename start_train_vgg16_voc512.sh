nohup python -u train_refinedet.py --dataset voc --input_size 512 --batch_size 32 --network vgg16 --basenet vgg16_reducedfc.pth --save_folder "weights/vgg16" > vgg16_voc512_nohup.out 2>&1 &
