export OPENBLAS_NUM_THREADS=1
export DETECTRON2_DATASETS=/media/dataset1/SSIS

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
--config-file ./configs/cityscapes/instance-segmentation/maskformer2_R50_bs16_90k.yaml \
--num-gpus ${ngpus} \
--dist-url 'auto' \
OUTPUT_DIR ./outputs/cityscapes_${percentage}/STUDENT \
SOLVER.IMS_PER_BATCH 16 \
SOLVER.CHECKPOINT_PERIOD 5000 \
SSL.TEACHER_CKPT ${teacher_ckpt} \
SSL.TRAIN_SSL True \
SSL.USE_SAM True \
SSL.SAM_CKPT_DIR ./model_weights \
TEST.EVAL_PERIOD 5000 \
SSL.PERCENTAGE ${percentage} \
$opts