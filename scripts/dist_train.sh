#!/usr/bin/env bash
GPUS=$1
CONFIG=$2
PORT=${PORT:-4830}

function check_port() {
    local port=$1
    nc -z localhost $port
    return $?
}

if [ $# -lt 2 ] ;then
    echo "usage:"
    echo "./scripts/dist_train.sh [number of gpu] [path to option file]"
    exit
fi

while check_port $PORT; do
    PORT=$((PORT + 1))
done

TORCH_VERSION=$(python -c "import torch; print(torch.__version__)")

if [[ "$TORCH_VERSION" < "1.9.0" ]]; then
    PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
    python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$PORT \
        basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}
else
    PYTHONPATH="$(dirname $0)/..:${PYTHONPATH}" \
    torchrun --nproc_per_node=$GPUS --master_port=$PORT \
        basicsr/train.py -opt $CONFIG --launcher pytorch ${@:3}
fi