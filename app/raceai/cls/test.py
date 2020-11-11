#!/usr/bin/python3
# -*- coding: utf-8 -*-

import tempfile
import torch
from raceai.utils.error import catch_error
from raceai.utils.misc import race_load_class


@catch_error
def image_classifier_test(cfg):
    with tempfile.TemporaryDirectory() as tmp_dir:
        # Data
        data_loader = race_load_class(cfg.data.class_name)(tmp_dir, cfg.data.params)
        test_loader = data_loader.get_testloader()

        # Model
        test_model = race_load_class(cfg.model.class_name)(cfg.model.params)
        if cfg.general.use_gpu:
            test_model.cuda()

        # Test
        test_model.eval()
        with torch.no_grad():
            results = []
            for inputs in test_loader:
                if cfg.general.use_gpu:
                    inputs = inputs.cuda()
                outputs = test_model(inputs)
                results.extend(outputs.argmax(dim=1).cpu().numpy().tolist())
            return {'errno': 0, 'result': results}
