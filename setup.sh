# Set up dataset root here.
DATA_ROOT=''
CIFAR10_DATA_DIR="$DATA_ROOT/cifar-10"
CIFAR100_DATA_DIR="$DATA_ROOT/cifar-100"
IMAGENET_DATA_DIR="$DATA_ROOT/imagenet"

# Set up training logs path here.
LOGS_DIR=''

# Set up model save path here.
SAVE_DIR=''

# Put 'yes' here to download CIFAR and ImageNet datasets.
DOWNLOAD_CIFAR='no'
DOWNLOAD_IMAGENET='no'

mkdir -p data
if [ $DOWNLOAD_CIFAR = 'yes' ] then
  cd tools
  ./download_cifar.sh $DATA_ROOT
  cd ..
fi
if [ $CIFAR10_DATA_DIR ] then
  ln -s $CIFAR10_DATA_DIR "data/cifar-10"
fi
if [ $CIFAR10_DATA_DIR ] then
  ln -s $CIFAR100_DATA_DIR "data/cifar-100"
fi
if [ $IMAGENET_DATA_DIR ] then
  ln -s $IMAGENET_DATA_DIR "data/imagenet"
fi
if [ $DOWNLOAD_IMAGENET = 'yes' ] then
  cd tools
  ./download_and_preprocess_imagenet.sh $IMAGENET_DATA_DIR
  cd ..
fi

mkdir -p "logs"
if [ $LOGS_DIR ] then
  ln -s LOGS_DIR "logs/public"
else
  mkdir -p "logs/public"
fi

if [ $SAVE_DIR ] then
  ln -s $SAVE_DIR "results"
else
  mkdir -p "results"
fi
