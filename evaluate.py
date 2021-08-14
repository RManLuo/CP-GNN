#!/usr/bin/env python 
# -*- coding: utf-8 -*-
# @Time    : 2020/8/11 11:11
# @Author  : Raymound luo
# @Mail    : luolinhao1998@gmail.com
# @File    : evaluate.py
# @Software: PyCharm
# @Describe:

from models import ContextGNN
from utils import load_data, evaluate, load_latest_model, save_attention_matrix, generate_attention_heat_map, \
    save_config
import torch
import importlib
import os
import argparse


def evaluate_task(config, checkpoint_path=None):
    dataloader = load_data(config.data_config)
    hg = dataloader.heter_graph
    if config.data_config['dataset'] in ['AIFB', 'AM', 'BGS', 'MUTAG']:
        config.data_config['primary_type'] = dataloader.predict_category
        config.model_config['primary_type'] = dataloader.predict_category
    if not checkpoint_path:
        model = ContextGNN(hg, config.model_config)
        model = load_latest_model(config.train_config['checkpoint_path'], model)
    else:
        config_path = os.path.join(checkpoint_path, 'config')
        config_path = os.path.relpath(config_path)
        config_file = config_path.replace(os.sep, '.')
        model_path = os.path.join(checkpoint_path, 'model.pth')
        config = importlib.import_module(config_file)
        model = ContextGNN(hg, config.model_config)
        model.load_state_dict(torch.load(model_path))
    p_emb = model.primary_emb.weight.detach().cpu().numpy()
    CF_data = dataloader.load_classification_data()
    # LP_data = dataloader.load_links_prediction_data()
    result_save_path = evaluate(p_emb, CF_data, None, method=config.evaluate_config['method'],  metric=config.data_config['task'], save_result=True,
                                result_path=config.evaluate_config['result_path'],
                                random_state=config.evaluate_config['random_state'],
                                max_iter=config.evaluate_config['max_iter'], n_jobs=config.evaluate_config['n_jobs'])
    if result_save_path:
        save_config(config, result_save_path)
        model_save_path = os.path.join(result_save_path, "model.pth")
        torch.save(model.state_dict(), model_save_path)
        attention_matrix_path = save_attention_matrix(model, result_save_path, config.data_config['K_length'])
        if attention_matrix_path and config.evaluate_config['save_heat_map']:
            generate_attention_heat_map(hg.ntypes, attention_matrix_path)


if __name__ == "__main__":
    import config

    parser = argparse.ArgumentParser(description='Which checkpoint to load?')
    parser.add_argument('-path', default=None, type=str, help='checkpoint path')
    args = parser.parse_args()
    evaluate_task(config, args.path)
