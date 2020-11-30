#!/usr/bin/python3
# -*- coding: utf-8 -*-

# @file trainning.py
# @brief
# @author QRS
# @version 1.0
# @date 2020-11-27 13:05

import argparse
import json

from omegaconf import OmegaConf
from raceai.utils.registrable import FunctionRegister
from raceai.utils.error import catch_error
from raceai.utils.misc import race_load_class

from raceai.runner.pl import PlClassifier


@catch_error
@FunctionRegister.register('cls.training.pl')
def pl_classifier_fit(cfg):
    # Data
    train_loader_class = race_load_class(cfg.data.train.class_name)
    valid_loader_class = race_load_class(cfg.data.valid.class_name)
    train_loader = train_loader_class(cfg.data.train.params).get()
    valid_loader = valid_loader_class(cfg.data.valid.params).get()

    # Model
    model = race_load_class(cfg.model.class_name)(cfg.model.params)

    # Solver
    optimizer_class = race_load_class(cfg.solver.optimizer.class_name)
    scheduler_class = race_load_class(cfg.solver.scheduler.class_name)
    optimizer = optimizer_class(model.parameters(), **cfg.solver.optimizer.params)
    scheduler = scheduler_class(optimizer, **cfg.solver.scheduler.params)

    # Trainer
    logger = False
    if 'logger' in cfg.solver.trainer.params:
        logger_class = race_load_class(cfg.solver.trainer.params.logger.class_name)
        logger = logger_class(**cfg.solver.trainer.params.logger.params)
    callbacks = []
    if 'callbacks' in cfg.solver.trainer.params:
        for cb in cfg.solver.trainer.params.callbacks:
            callbacks.append(race_load_class(cb.class_name)(**cb.params))
    trainer_class = race_load_class(cfg.solver.trainer.class_name)
    trainer = trainer_class(logger, callbacks, **cfg.solver.trainer.params.general)

    ## runner.fit
    classifier = PlClassifier(trainer, model, optimizer, scheduler)
    classifier.fit(train_loader, valid_loader)

    ## runner.test
    if 'test' in cfg.data:
        test_loader_class = race_load_class(cfg.data.test.class_name)
        test_loader = test_loader_class(cfg.data.test.params).get()
        result = classifier.test(test_loader)
    return {'errno': 0, 'result': result}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            '--config',
            default='0.0.0.0',
            type=str,
            dest='config',
            help="json configure file")

    args = parser.parse_args()

    with open(args.config, 'r') as f:
        config = json.load(f)
        runner = FunctionRegister.get_runner(config['task'])
        cfg = OmegaConf.create(config['cfg'])

    print(runner(cfg))
