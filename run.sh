########################################
#### Configurations for Pretraining ####
########################################
PORT_NUM=25606
GPU_num=4 ## 4GPU
DEVICES=0,1,2,3
net_name=CLeVER ##dino, CLeVER
backbone_name=vit_tiny ## resnet18, resnet50, vit_tiny, vit_small, vssm2-vmambav2_tiny_224
EPOCH=200
DATASET=IN100 ##Imagenet, IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("65536" "" "" "" "_reg0.001") ## ("out_dim: 410/2048/16384/65536" "NA" "NA" "NA" "_reg0.001/blank")
HP1=("0.8") ## hyperparameter for separation ratio of representations of IR and EF (default 0.8).
BATCH=("128") ## 128 for 4*GPUs / 256 for 2*GPUs
AUG_TYPE=("aug1_2") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0") ## The coefficient of Orthogonal loss (default 1.0)
#### Configurations for Linear Probe
GPU_num_LN=2
DEVICES_LN=0,1
BATCH_LN=128
OTHER_LINEAR_PARA=("sgd" "0.001" "" "100")
#### Configurations for Evaluation after Linear Probe 
GPU_num_EVAL=1
DEVICES_EVAL=0
########################
Result_dir=<path to output path>
PARA_dir=${DATASET}_ep${EPOCH}/${net_name}_${backbone_name}/${AUG_TYPE[0]}/${BATCH[0]}/${HP1[0]}/
TRIAL_name=${net_name}_${backbone_name}_${BATCH[0]}_${HP1[0]}_${OTHER_PARA[0]}_${SEP_LAMBD[0]}${OTHER_PARA[4]}
echo ${PARA_dir} ${TRIAL_name}
mkdir -p ${Result_dir}/${PARA_dir}/${TRIAL_name}/

#### Code for Pre-training
CUDA_VISIBLE_DEVICES=${DEVICES} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num} --master_port=${PORT_NUM} main_dino.py --net ${net_name} --arch ${backbone_name} --data_path ${DATASET_PATH}/train --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --epochs ${EPOCH} --batch_size_per_gpu ${BATCH[0]} --aug ${AUG_TYPE[0]} --hp1 ${HP1[0]} --DVR_out_dim ${OTHER_PARA[0]} --sep_lambd ${SEP_LAMBD[0]} --reg_lambd ${OTHER_PARA[4]} --saveckp_freq 100 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_pretrain_${TRIAL_name}.txt

#### Code for Linear Probe
## For dino and CLeVER
## linear probe using full representation
CUDA_VISIBLE_DEVICES=${DEVICES_LN} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_LN} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --batch_size_per_gpu ${BATCH_LN} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[0]} --else_part ALL --n_last_blocks 4 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_linear_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt

## For CLeVER only
## linear probe using only invariant representation (IR)
CUDA_VISIBLE_DEVICES=${DEVICES_LN} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_LN} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --batch_size_per_gpu ${BATCH_LN} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[0]} --else_part main_part --n_last_blocks 4 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_linear_hp${HP1[0]}_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt

## linear probe using only equivariant factor (EF)
CUDA_VISIBLE_DEVICES=${DEVICES_LN} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_LN} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --batch_size_per_gpu ${BATCH_LN} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[0]} --else_part else_part --n_last_blocks 4 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_linear_hp${HP1[0]}_else_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt ## --n_last_blocks 1 --avgpool_patchtokens true

#### Performance evaluation of perturbed input images
TEST_AUG_TYPE=("basic" "aug1" "aug1_2" "aug1_4_2")
for test_aug in {0..3..1}
do
## For dino and CLeVER
## evaluation using full representation
CUDA_VISIBLE_DEVICES=${DEVICES_EVAL} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_EVAL} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[0]} --else_part ALL --n_last_blocks 4 --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --evaluate --final_eval_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint_linear_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.pth.tar > ${Result_dir}/${PARA_dir}/${TRIAL_name}/only_eval_${TEST_AUG_TYPE[${test_aug}]}_linear_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt

## For CLeVER only
## evaluation using only invariant representation (IR)
CUDA_VISIBLE_DEVICES=${DEVICES_EVAL} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_EVAL} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[0]} --else_part main_part --n_last_blocks 4 --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --evaluate --final_eval_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint_linear_hp${HP1[0]}_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.pth.tar > ${Result_dir}/${PARA_dir}/${TRIAL_name}/only_eval_${TEST_AUG_TYPE[${test_aug}]}_linear_hp${HP1[0]}_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt

## evaluation using only equivariant factor (EF)
CUDA_VISIBLE_DEVICES=${DEVICES_EVAL} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_EVAL} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[0]} --else_part else_part --n_last_blocks 4 --test_aug_type ${TEST_AUG_TYPE[${test_aug}]} --evaluate --final_eval_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint_linear_hp${HP1[0]}_else_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.pth.tar > ${Result_dir}/${PARA_dir}/${TRIAL_name}/only_eval_${TEST_AUG_TYPE[${test_aug}]}_linear_hp${HP1[0]}_else_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt
done



