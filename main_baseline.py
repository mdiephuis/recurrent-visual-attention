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
import torchvision

from trainer import Trainer
from config import get_config
from utils import prepare_dirs, save_config
from datasets.utils import normalize_images, resize_lambda
from datasets.loader import get_loader
import argparse
from functools import partial
import pprint


def main(config):

    pp = pprint.PrettyPrinter(indent=4)
    pp.pprint(config)

    # ensure directories are setup
    prepare_dirs(config)

    # ensure reproducibility
    torch.manual_seed(config.random_seed)
    kwargs = {}
    if config.cuda:
        torch.cuda.manual_seed(config.random_seed)
        # kwargs = {'num_workers': 1, 'pin_memory': True}

    transform = torchvision.transforms.Resize(size=(config.height, config.width))
    dl = get_loader(config, transform=[transform], **vars(config))

    # instantiate trainer
    trainer = Trainer(config, dl)

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
