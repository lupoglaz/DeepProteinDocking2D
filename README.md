# DeepProteinDocking2D
Predicting the physical interaction of proteins is a cornerstone problem in computational biology. New learning-based algorithms are typically trained end-to-end on protein structures extracted from the Protein Data Bank. However, these training datasets tend to be large and difficult to use for prototyping and, unlike image or natural language datasets, they are not easily interpretable by non-experts.

In this paper we propose Dock2D-IP and Dock2D-FI, two toy datasets that can be used to select algorithms predicting protein-protein interactions (or any other type of molecular interactions). Using two-dimensional shapes as input, each example from Dock2D-FI describes the fact of interaction (FI) between two shapes and each example from Dock2D-IP describes the interaction pose (IP) of two shapes known to interact.

We propose baselines that represent different approaches to the problem and demonstrate the potential for transfer learning across the IP prediction and FI prediction tasks.

# Dataset 
The generated dataset is available here:
https://github.com/lupoglaz/DeepProteinDocking2D/releases/download/Dataset/toy_dataset.tar.gz

The main dataset generation script is *generate_dataset.py*. Example usage:
python generate_dataset.py -test -gen

Then to visualize different metrics presented in the paper you can use:
python generate_dataset.py -test -vis

Beware that this branch is constantly modified and for example right now to generate the dataset from scratch you have to uncomment these lines:
https://github.com/lupoglaz/DeepProteinDocking2D/blob/8ab5afeb9b1cd7cf08b7fd8cc9515b561b3ccd1a/generate_dataset.py#L65
https://github.com/lupoglaz/DeepProteinDocking2D/blob/8ab5afeb9b1cd7cf08b7fd8cc9515b561b3ccd1a/generate_dataset.py#L67

# Baselines
To run all the experiments, use *run.sh*, just uncomment the lines corresponding to the baseline of interest. However, we do not mention Energy-based models in the MLSB publication. We ask you to not pay attention to the experiment that are not published yet. 

The logs are written to the *data_dir* argument of the scripts. To visualize them please use tennsorboard.
