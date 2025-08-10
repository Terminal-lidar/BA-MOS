#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import os
import random
import numpy as np
import torch
import __init__ as booger

from datetime import datetime
from utils.utils import *
from modules.trainer_refine import TrainerRefine


def set_seed(seed=1024):
    random.seed(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.


if __name__ == '__main__':
    parser = get_args(flags="train")
    FLAGS, unparsed = parser.parse_known_args()
    FLAGS.log = os.path.join(FLAGS.log, datetime.now().strftime("%Y-%-m-%d-%H:%M") + FLAGS.name)
    
    # open arch / data config file
    ARCH = load_yaml(FLAGS.arch_cfg)
    DATA = load_yaml(FLAGS.data_cfg)

    make_logdir(FLAGS=FLAGS, resume_train=False) # create log folder
    check_pretrained_dir(FLAGS.pretrained) # does model folder exist?
    backup_to_logdir(FLAGS=FLAGS, pretrain_model=True) # backup code and config files to logdir

    set_seed()
    # create trainer and start the training
    trainer = TrainerRefine(ARCH, DATA, FLAGS.dataset, FLAGS.log, FLAGS.pretrained)

    print("----------")
    print("INTERFACE:")
    print("  dataset:", FLAGS.dataset)
    print("  arch_cfg:", FLAGS.arch_cfg)
    print("  data_cfg:", FLAGS.data_cfg)
    print("  log:", FLAGS.log)
    print("  pretrained:", FLAGS.pretrained)
    print("  Augmentation for residual: {}, interval in validation: {}".format(ARCH["train"]["residual_aug"],
                                                                               ARCH["train"]["valid_residual_delta_t"]))
    print("----------\n")

    trainer.train()
