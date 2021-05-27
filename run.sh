#Brute-force docking
# python train_docking.py -experiment BFDocking -train -docker -num_epochs 10 -batch_size 8 -gpu 0

#Brute-force interaction
# python train_interaction.py -experiment BFInteraction -train -docker -batch_size 12 -num_epochs 1

#Energy-based docking
python train_docking.py -experiment EBMDocking -train -ebm -num_epochs 100 -batch_size 32 -gpu 0
python results.py -experiment EBMDocking -docking -max_epoch 100

#ResNet docking
# python train_docking.py -experiment CNNDocking -train -resnet -num_epochs 100 -batch_size 32 -gpu 1

#ResNet interaction
# python train_interaction.py -experiment CNNDocking -train -resnet -num_epochs 100 -batch_size 32 -gpu 1