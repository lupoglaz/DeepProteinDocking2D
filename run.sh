#Brute-force docking
# python train_docking.py -experiment BFDocking -docker -num_epochs 30 -batch_size 11 -gpu 0
# python results.py -experiment BFDocking -docking -max_epoch 10


#Brute-force interaction
# python train_interaction.py -experiment BFInteraction -train -docker -pretrain BFDocking -batch_size 8 -num_epochs 100
python train_interaction.py -experiment BFInteraction -train -docker -batch_size 8 -num_epochs 100
# python results.py -experiment BFInteraction -interaction -max_epoch 10


#Energy-based docking
# python train_docking.py -experiment EBMDocking -ebm -num_epochs 100 -batch_size 32 -gpu 0
# python results.py -experiment EBMDocking -docking -max_epoch 100
# python train_docking.py -experiment EBMDocking -test -ebm -gpu 0

# python train_docking.py -experiment EBMDocking_GlobalStep1 -train -ebm -num_epochs 100 -batch_size 32 -gpu 0 -no_global_step
# python results.py -experiment EBMDocking_GlobalStep1 -docking -max_epoch 100
# python train_docking.py -experiment EBMDocking_GlobalStep -test -ebm -gpu 0
# python train_docking.py -experiment EBMDocking_GlobalStep1 -test -ebm -gpu 0

# python train_docking.py -experiment EBMDocking_AddPos1 -train -ebm -num_epochs 100 -batch_size 32 -gpu 0 -no_pos_samples
# python results.py -experiment EBMDocking_AddPos1 -docking -max_epoch 100
# python train_docking.py -experiment EBMDocking_AddPos -test -ebm -gpu 0
# python train_docking.py -experiment EBMDocking_AddPos1 -test -ebm -gpu 0

# python train_docking.py -experiment EBMDocking_Default -train -ebm -num_epochs 100 -batch_size 32 -gpu 0 -default
# python results.py -experiment EBMDocking_Default -docking -max_epoch 100
# python train_docking.py -experiment EBMDocking_Default -test -ebm -gpu 0


#ResNet docking
# python train_docking.py -experiment CNNDocking -train -resnet -num_epochs 100 -batch_size 32 -gpu 1

#ResNet interaction
# python train_interaction.py -experiment CNNInteraction -train -resnet -num_epochs 30 -batch_size 32 -gpu 1
# python train_interaction.py -experiment CNNInteraction -test -resnet -gpu 0