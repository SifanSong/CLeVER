#############################
#### semi and downstream ####
#############################
PORT_NUM=25712
GPU_num=1
DEVICES=0
net_name=CLeVER ## directly change CLeVER to dino to conduct dino experiments
backbone_name=vit_tiny ## resnet18, resnet50, vit_tiny, vit_small, vssm2-vmambav2_tiny_224
EPOCH=200
DATASET=IN100 ##Imagenet, IN100
DATASET_PATH=<path to imagenet-100 or imagenet>
OTHER_PARA=("65536" "" "" "" "_reg0.001") ## ("out_dim: 410/2048/16384/65536" "NA" "NA" "NA" "_reg0.001/blank")
HP1=("0.8") ## hyperparameter for separation ratio of representations of IR and EF (default 0.8).
BATCH=("128") ## identical to pretraining
AUG_TYPE=("aug1")
SEP_LAMBD=("1.0")
OTHER_LINEAR_PARA=("sgd" "0.001" "" "100") # "sgd" "0.001" "256" "100" / "sgd" "30" "256" "200"
####################
DS_DATA=("CUB200" "Flowers102" "Food101" "OxfordIIITPet" "CUB200" "Flowers102" "Food101" "OxfordIIITPet" ${DATASET} ${DATASET})
DS_MODE=("linear" "linear" "linear" "linear" "ds_ft" "ds_ft" "ds_ft" "ds_ft" "semi_1" "semi_10")
DS_BATCH=("256" "256" "256" "256" "256" "256" "256" "256" "256" "256")
DS_EPOCH=("200" "200" "200" "200" "200" "200" "200" "200" "200" "200")
DS_WEIGHT_DECAY=("0.0" "0.0" "0.0" "0.0" "0.0001" "0.0001" "0.0001" "0.0001" "0.0001" "0.0001")
LR_CLASSIFER=("0.001" "0.001" "0.001" "0.001" "0.001" "0.001" "0.001" "0.001" "0.001" "0.001")
LR_BACKBONE=("0" "0" "0" "0" "0.001" "0.001" "0.001" "0.001" "0.001" "0.001")
DS_AUG=("_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1" "_li_aug1")
DATASET_PATH_LIST=("/home/sifan2/Datasets/CUB_200_2011/" ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH} "/home/sifan2/Datasets/CUB_200_2011/" ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH} ${DATASET_PATH})

for ds_num in {0..9..1}
do
	Result_dir=<path to output path>
	PARA_dir=${DATASET}_ep${EPOCH}/${net_name}_${backbone_name}/${AUG_TYPE[0]}/${BATCH[0]}/${HP1[0]}/
	TRIAL_name=${net_name}_${backbone_name}_${BATCH[0]}_${HP1[0]}_${OTHER_PARA[0]}_${SEP_LAMBD[0]}${OTHER_PARA[4]}
	FILE_name=lr${LR_CLASSIFER[${ds_num}]}_lrbk${LR_BACKBONE[${ds_num}]}_ep${DS_EPOCH[${ds_num}]}_bs${DS_BATCH[${ds_num}]}${DS_AUG[${ds_num}]}
	mkdir -p ${Result_dir}/${PARA_dir}/${TRIAL_name}/ds/

	## For dino and CLeVER
	## ds using full representation
	CUDA_VISIBLE_DEVICES=${DEVICES} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num} --master_port=${PORT_NUM} eval_linear_ds.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --trial_name ${TRIAL_name} --ds_mode ${DS_MODE[${ds_num}]} --data_path ${DATASET_PATH_LIST[${ds_num}]} --dataset_type ${DS_DATA[${ds_num}]} --lr ${LR_CLASSIFER[${ds_num}]} --backbone_lr ${LR_BACKBONE[${ds_num}]} --weight_decay ${DS_WEIGHT_DECAY[${ds_num}]} --epochs ${DS_EPOCH[${ds_num}]} --batch_size_per_gpu ${DS_BATCH[${ds_num}]} --hp1 ${HP1[0]} --else_part ALL --n_last_blocks 4 --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.pth.tar > ${Result_dir}/${PARA_dir}/${TRIAL_name}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.txt

	## ds using only invariant representation (IR)
	CUDA_VISIBLE_DEVICES=${DEVICES} OMP_NUM_THREADS=1 python -m torch.distributed.launch --nproc_per_node=${GPU_num} --master_port=${PORT_NUM} eval_linear_ds.py --net ${net_name} --arch ${backbone_name} --pretrained_weights ${Result_dir}/${PARA_dir}/${TRIAL_name}/checkpoint.pth --trial_name ${TRIAL_name} --ds_mode ${DS_MODE[${ds_num}]} --data_path ${DATASET_PATH_LIST[${ds_num}]} --dataset_type ${DS_DATA[${ds_num}]} --lr ${LR_CLASSIFER[${ds_num}]} --backbone_lr ${LR_BACKBONE[${ds_num}]} --weight_decay ${DS_WEIGHT_DECAY[${ds_num}]} --epochs ${DS_EPOCH[${ds_num}]} --batch_size_per_gpu ${DS_BATCH[${ds_num}]} --hp1 ${HP1[0]} --else_part main_part --n_last_blocks 4 --output_dir ${Result_dir}/${PARA_dir}/${TRIAL_name}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear_hp${HP1[0]}_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.pth.tar > ${Result_dir}/${PARA_dir}/${TRIAL_name}/ds/ds_${DS_MODE[${ds_num}]}_${DS_DATA[${ds_num}]}_linear_hp${HP1[0]}_${TRIAL_name}_${OTHER_LINEAR_PARA[0]}_${FILE_name}.txt

done
