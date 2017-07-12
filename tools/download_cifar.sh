OUTPUT_DIR="${1%/}"
CIFAR10_URL="https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR100_URL="https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz"
CIFAR10_DIR="$OUTPUT_DIR/cifar-10"
CIFAR100_DIR="$OUTPUT_DIR/cifar-100"

echo "Downloading CIFAR-10"
mkdir -p $CIFAR10_DIR
wget $CIFAR10_URL -O "$CIFAR10_DIR/cifar-10.tar.gz"
tar xzf "$CIFAR10_DIR/cifar-10.tar.gz" -C $CIFAR10_DIR

echo "Downloading CIFAR-100"
mkdir -p $CIFAR100_DIR
wget $CIFAR100_URL -O "$CIFAR100_DIR/cifar-100.tar.gz"
tar xzf "$CIFAR100_DIR/cifar-100.tar.gz" -C $CIFAR100_DIR
