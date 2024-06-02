import os

import pandas as pd


def check_model_method(model, model_dict, libraries):
    # get model_dict part in which library is in libraries
    for lib in libraries:
        if lib in model_dict['library']:
            # take part of dict in which model_dict['library'] is in lib
            # get model_dict part in which model is in library
            for model_part in model_dict['method']:
                if model_part == model and lib in model_dict['library']:
                    return True
    return False


def load_model_dict():
    dict = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../obj_dictionaries/models.csv')).to_dict(
        orient='list')
    return dict


def load_tensor_operations_dict():
    dict = pd.read_csv(os.path.join(os.path.dirname(__file__), '../../obj_dictionaries/tensors.csv'))
    dict = dict[dict['number_of_tensors_input'] > 1]
    return dict.to_dict(orient='list')
