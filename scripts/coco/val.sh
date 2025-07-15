ngpus=$1
model_weights=$2

if [ -z $ngpus ]
then
    echo "Number of gpus not specified"
    exit 0
fi

if [ -z $model_weights ]
then
    echo "Model weights not specified"
    exit 0
fi

shift 2
opts=${@}

python3 train_net.py \
--num-gpus ${ngpus} \
--config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
--dist-url 'auto' \
--eval-only \
MODEL.WEIGHTS ${model_weights} \
$opts