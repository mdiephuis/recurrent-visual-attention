''' 
Generate baseline stats for (high res) cluttered MNIST

todo:
* do argparse nicer
* pass the image preproc nicer, upsamle, normalize
* visualize image samples-> integrate grapher and check out show_sample in data_loader.py
* 
*
'''

import torch

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from datasets.utils import normalize_images, resize_lambda, normalize_images_fixed
# from data_loader import get_test_loader, get_train_valid_loader
from datasets.loader import get_loader
import argparse

# add helper.utils number_of_parameters -> print
# check number of hops

def main(config):

    # ensure directories are setup
    prepare_dirs(config)

    # Hack
    parser = argparse.ArgumentParser(description='')
    args = parser.parse_args()
    args.task = 'HRes_clutter'
    args.data_dir = config.data_dir
    args.batch_size = config.batch_size
    args.random_seed = config.random_seed
    args.valid_size = config.valid_size
    args.shuffle = config.shuffle
    args.cuda = config.use_gpu
    args.data_dir = 'data/cluttered_mnist'
    args.transform = [normalize_images_fixed]
    args.synthetic_upsample_size = 200
    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.use_gpu:
        torch.cuda.manual_seed(config.random_seed)
        kwargs = {'num_workers': 1, 'pin_memory': True}

    # instantiate data loaders
    # todo: pass normalization
    # todo: pass upsampling
    data_loader = []
    if config.is_train:
        dl = get_loader(args)
        data_loader= [dl.train_loader, dl.test_loader]
    else:
        dl = get_loader(args)
        data_loader= [dl.train_loader, dl.test_loader]

    # instantiate trainer
    trainer = Trainer(config, data_loader)

    # either train
    if config.is_train:
        save_config(config)
        trainer.train()

    # or load a pretrained model and test
    else:
        trainer.test()


if __name__ == '__main__':
    config, unparsed = get_config()
    main(config)
