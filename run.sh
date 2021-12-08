#Brute-force docking
# python train_docking.py -experiment BFDocking -docker -num_epochs 30 -batch_size 11 -gpu 0

#Energy-based docking
# python train_docking.py -experiment EBMDocking -ebm -num_epochs 100 -batch_size 32 -gpu 0
# python train_docking.py -experiment EBMDocking_GlobalStep1 -ebm -num_epochs 100 -batch_size 32 -gpu 0 -no_global_step
# python train_docking.py -experiment EBMDocking_AddPos1 -ebm -num_epochs 100 -batch_size 32 -gpu 0 -no_pos_samples
# python train_docking.py -experiment EBMDocking_Default -ebm -num_epochs 100 -batch_size 32 -gpu 0 -default


#ResNet docking
# python train_docking.py -experiment CNNDocking -resnet -num_epochs 100 -batch_size 32 -gpu 0


#Brute-force interaction
# python train_interaction.py -experiment BFInteraction -docker -pretrain BFDocking -batch_size 8 -num_epochs 10
# python train_interaction.py -experiment BFInteraction_repl -docker -batch_size 8 -num_epochs 20

#ResNet interaction
# python train_interaction.py -experiment CNNInteraction -resnet -num_epochs 30 -batch_size 32
