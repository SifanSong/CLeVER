########################
PORT_NUM=25606
GPU_num=4 ## 1GPU or 2GPU
DEVICES=0,1,2,3
GPU_num_LN=2
DEVICES_LN=2,3
BATCH_LN=128
GPU_num_EVAL=1
DEVICES_EVAL=0
net_name=CLeVER ##dino, CLeVER
backbone_name=vit_tiny ## vit_tiny, vit_small, vssm2-vmambav2_tiny_224, resnet18, resnet50
EPOCH=50
DATASET=IN100 ##Imagenet, IN100
DATASET_PATH=/home/sifan2/ImageNet/imagenet-100
OTHER_PARA=("65536" "" "" "" "_reg0.001") ## ("DVR_out_dim: 410/2048/16384/65536" "NA" "NA" "NA" "_reg0.001/blank")
HP1=("0.8") ## hyperparameter for separation ratio of representations.
BATCH=("128") ## 128 for 4*GPUs / 256 for 2*GPUs
AUG_TYPE=("aug1_2") ## different augmentation types: aug1=BAug, aug1_2=CAug, aug1_4_2=CAug+ (identical to the manuscript)
SEP_LAMBD=("1.0")
OTHER_LINEAR_PARA=("sgd" "0.001" "" "100")
########################
for rep in {0..0..1}
do
for batch in {0..0..1}
do
for hp1 in {0..0..1}
do
for aug_type in {0..0..1}
do
	Result_dir=/home/sifan2/tmp/CLeVER/Results/
	PARA_dir=${DATASET}_ep${EPOCH}/${net_name}_${backbone_name}/${AUG_TYPE[${aug_type}]}/${BATCH[${batch}]}/${HP1[${hp1}]}/
	TRIAL_name=${net_name}_${backbone_name}_${BATCH[${batch}]}_${HP1[${hp1}]}_${OTHER_PARA[0]}_${SEP_LAMBD[0]}${OTHER_PARA[4]}_rep${rep}
	echo ${PARA_dir} ${TRIAL_name}
	## train
	mkdir -p ${Result_dir}/${PARA_dir}/${TRIAL_name}/

	CUDA_VISIBLE_DEVICES=${DEVICES} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num} --master_port=${PORT_NUM} main_dino.py --net ${net_name} --arch ${backbone_name} --data_path ${DATASET_PATH}/train --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --epochs ${EPOCH} --batch_size_per_gpu ${BATCH[${batch}]} --aug ${AUG_TYPE[${aug_type}]} --hp1 ${HP1[${hp1}]} --DVR_out_dim ${OTHER_PARA[0]} --sep_lambd ${SEP_LAMBD[0]} --reg_lambd ${OTHER_PARA[4]} --saveckp_freq 100 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_pretrain_${TRIAL_name}.txt && \
	CUDA_VISIBLE_DEVICES=${DEVICES_LN} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_LN} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --batch_size_per_gpu ${BATCH_LN} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[${hp1}]} --else_part ALL --n_last_blocks 4 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_linear_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt && \
	CUDA_VISIBLE_DEVICES=${DEVICES_LN} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_LN} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --batch_size_per_gpu ${BATCH_LN} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[${hp1}]} --else_part main_part --n_last_blocks 4 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_linear_hp${HP1[${hp1}]}_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt && \
	CUDA_VISIBLE_DEVICES=${DEVICES_LN} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num_LN} --master_port=${PORT_NUM} eval_linear.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ --trial_name ${TRIAL_name} --data_path ${DATASET_PATH} --dataset_type ${DATASET} --batch_size_per_gpu ${BATCH_LN} --lr ${OTHER_LINEAR_PARA[1]} --epochs ${OTHER_LINEAR_PARA[3]} --hp1 ${HP1[${hp1}]} --else_part else_part --n_last_blocks 4 > ${Result_dir}/${PARA_dir}/${TRIAL_name}/result_linear_hp${HP1[${hp1}]}_else_${OTHER_LINEAR_PARA[1]}_${OTHER_LINEAR_PARA[3]}_${TRIAL_name}.txt ## --n_last_blocks 1 --avgpool_patchtokens true
done
done
done
done
