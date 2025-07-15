ngpus=$1
percentage=$2
teacher_ckpt=$3

if [ -z $ngpus ]
then
    echo "Number of gpus not specified"
    exit 0
fi

if [ -z $percentage ]
then
    echo "Labeled data percentage not specified"
    exit 0
fi

if [ -z $teacher_ckpt ]
then
    echo "Teacher checkpoint not specified"q
    exit 0
fi

shift 3
opts=${@}

python3 train_net.py \
--config-file ./configs/coco/instance-segmentation/maskformer2_R50_bs16_50ep.yaml \
--num-gpus ${ngpus} \
--dist-url 'auto' \
OUTPUT_DIR ./outputs/coco_${percentage}/STUDENT \
SOLVER.IMS_PER_BATCH 12 \
SOLVER.CHECKPOINT_PERIOD 7000 \
SSL.TEACHER_CKPT ${teacher_ckpt} \
SSL.BURNIN_ITER 25000 \
SSL.TRAIN_SSL True \
SSL.USE_SAM True \
SSL.AUG_STATIC False \
SSL.SAM_CKPT_DIR ./model_weights \
TEST.EVAL_PERIOD 7000 \
SSL.PERCENTAGE ${percentage} \
$opts