import _pickle as pkl
from pathlib import Path
from DatasetGeneration import DockerGPU, ProteinPool, ParamDistribution, InteractionCriteria
import random

if __name__ == '__main__':
	params = ParamDistribution(
		alpha = [(0.8, 1), (0.9, 9), (0.95, 4)],
		num_points = [(20, 1), (30, 2), (50, 4), (80, 8), (100, 6)],
		overlap = [(0.10, 2), (0.20, 3), (0.30, 4), (0.50, 2), (0.60, 1)]
		)
	
	docker = DockerGPU(boundary_size=3, a00=3.0, a10=-0.1, a11=-0.5)
	#Structures
	# pool = ProteinPool.generate(num_proteins=500, params=params)
	# pool.save('DatasetGeneration/protein_pool_big.pkl')
	
	#Interaction
	pool = ProteinPool.load('DatasetGeneration/Data/protein_pool_big.pkl')
	pool.get_interactions(docker)
	pool.save('DatasetGeneration/protein_pool_big.pkl')
	
	#Docking dataset
	pool = ProteinPool.load('DatasetGeneration/Data/protein_pool_big.pkl')
	inter = InteractionCriteria(score_cutoff=-70, funnel_gap_cutoff=10)
	dataset_all = pool.extract_docking_dataset(docker, inter, max_num_samples=1100)
	print(f'Total data length {len(dataset_all)}')
	random.shuffle(dataset_all)
	with open('DatasetGeneration/docking_data_train.pkl', 'wb') as fout:
		pkl.dump(dataset_all[:1000], fout)
	with open('DatasetGeneration/docking_data_valid.pkl', 'wb') as fout:
		pkl.dump(dataset_all[1000:], fout)

	#Interaction dataset
	pool = ProteinPool.load('DatasetGeneration/Data/protein_pool_big.pkl')
	inter = InteractionCriteria(score_cutoff=-70, funnel_gap_cutoff=10)
	interactome_train = pool.extract_interactome_dataset(inter, ind_range=(0, 900))
	interactome_valid = pool.extract_interactome_dataset(inter, ind_range=(900, 1000))
	with open('DatasetGeneration/interaction_data_train.pkl', 'wb') as fout:
		pkl.dump(interactome_train, fout)
	with open('DatasetGeneration/interaction_data_valid.pkl', 'wb') as fout:
		pkl.dump(interactome_valid, fout)

	#Visualization
	pool = ProteinPool.load('DatasetGeneration/Data/protein_pool_big.pkl')
	pool.plot_interaction_dist(perc=90)
	pool.plot_interactions(docker, num_plots=20, type='best')
	pool.plot_interactions(docker, num_plots=20, type='worst')
	pool.plot_sample_funnels(docker, filename='funnels.png', range=[(-100, -80), (-70, -40), (-32, -20)])
	pool.plot_params()
	
	