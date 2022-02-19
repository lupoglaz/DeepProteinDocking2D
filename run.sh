#Brute-force docking
# python train_docking.py -data_dir Log -experiment BFDocking -docker -train -num_epochs 30 -batch_size 11 -gpu 0
# python train_docking.py -data_dir Log -experiment BFDocking -docker -test -gpu 0

#Energy-based docking
#python train_docking.py -data_dir Log -experiment EBMDocking -train -ebm -num_epochs 100 -batch_size 32 -gpu 0
#python train_docking.py -data_dir Log -experiment EBMDocking -test -ebm -gpu 0
# python train_docking.py -experiment EBMDocking_GlobalStep1 -ebm -num_epochs 100 -batch_size 32 -gpu 0 -no_global_step
# python train_docking.py -experiment EBMDocking_AddPos1 -ebm -num_epochs 100 -batch_size 32 -gpu 0 -no_pos_samples
# python train_docking.py -experiment EBMDocking_Default -ebm -num_epochs 100 -batch_size 32 -gpu 0 -default


#ResNet docking
# python train_docking.py -data_dir Log -experiment CNNDocking -resnet -train -num_epochs 100 -batch_size 32 -gpu 0
# python train_docking.py -data_dir Log -experiment CNNDocking -resnet -test -gpu 0

#Brute-force interaction
# python train_interaction.py -experiment BFInteraction -docker -train -batch_size 8 -num_epochs 10
# python train_docking.py -data_dir Log -experiment BFInteraction -docker -test -gpu 0

#python train_interaction.py -experiment BFInteraction_georgydefaults_batch8_10ep -docker -train -batch_size 8 -num_epochs 10
#python train_docking.py -data_dir Log -experiment BFInteraction -docker -test -gpu 0

#ResNet interaction
# python train_interaction.py -data_dir Log -experiment CNNInteraction -resnet -train -num_epochs 100 -batch_size 32
# python train_interaction.py -data_dir Log -experiment CNNInteraction -resnet -test -gpu 0
