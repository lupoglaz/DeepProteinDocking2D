import os 
import sys 
import argparse
from pathlib import Path
from Logger import Logger

if __name__=='__main__':
	parser = argparse.ArgumentParser(description='Train deep protein docking')
	parser.add_argument('-experiment', default='Debug', type=str)
	parser.add_argument('-inter', default=False, type=str)
	args = parser.parse_args()

	LOG_DIR = Path('Log')/Path(args.experiment)
	RES_DIR = Path('Results')/Path(args.experiment)
	RES_DIR.mkdir(parents=True, exist_ok=True)

	logger = Logger(LOG_DIR)
	if args.inter:
		logger.plot_losses_int(RES_DIR/Path("losses"), average_num=30)
	else:
		max_epoch = 100
		logger.plot_losses(RES_DIR/Path("losses"))
		logger.plot_dock(RES_DIR/Path("dock"), max_epoch=max_epoch)
		logger.plot_eval(RES_DIR/Path("eval_anim"), max_epoch=max_epoch)
	