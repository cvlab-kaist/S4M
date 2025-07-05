ngpus=$1
percentage=$2

if [ -z $ngpus ]
then
    echo "Number of gpus not specified!"
    exit 0
fi

if [ -z $percentage ]
then
    echo "Labeled data percentage not specified!"
    exit 0
fi

shift 2
opts=${@}

python3 train_net.py \
--num-gpus ${ngpus} \
--config-file ./configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml \
--dist-url 'auto' \
OUTPUT_DIR ./outputs/cityscapes_${percentage}/TEACHER \
SOLVER.CHECKPOINT_PERIOD 5000 \
SOLVER.IMS_PER_BATCH 4 \
SOLVER.BASE_LR 0.0001 \
SSL.TRAIN_SSL False \
SSL.PERCENTAGE ${percentage} \
SSL.USE_SAM True \
SSL.SAM_CKPT_DIR ./model_weights \
TEST.EVAL_PERIOD 5000 \
$opts