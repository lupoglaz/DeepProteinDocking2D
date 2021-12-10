#Sampling results for statistics in the paper
python train_interaction.py -experiment BFInteraction_2 -docker -batch_size 8 -num_epochs 10

#ResNet interaction
python train_interaction.py -experiment CNNInteraction_1 -resnet -num_epochs 100 -batch_size 32
python train_interaction.py -experiment CNNInteraction_2 -resnet -num_epochs 100 -batch_size 32

for i in 3 4 5
do
    #Brute-force docking
    python train_docking.py -experiment BFDocking_$i -docker -num_epochs 30 -batch_size 11 -gpu 0 
    #Energy-based docking
    python train_docking.py -experiment EBMDocking_$i -ebm -num_epochs 100 -batch_size 32 -gpu 0
    #ResNet docking
    python train_docking.py -experiment CNNDocking_$i -resnet -num_epochs 100 -batch_size 32 -gpu 0
    #ResNet interaction
    python train_interaction.py -experiment CNNInteraction_$i -resnet -num_epochs 100 -batch_size 32
    #Brute-force interaction
    python train_interaction.py -experiment BFInteraction_$i -docker -num_epochs 10 -batch_size 8
    
done