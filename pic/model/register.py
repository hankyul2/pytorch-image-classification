import sys
from inspect import signature, _empty
from typing import Iterable

_name_to_model = {}
_argument_of_model = {}

def register_model(fn):
    # 1. load config dict
    module = sys.modules[fn.__module__]
    config = getattr(module, 'model_config', None)

    if config is None:
        raise ValueError(f"please define model_config dictionary for {fn.__name__}")

    for model_name, model_config in config.items():
        if model_name in _name_to_model:
            raise ValueError(f"please change {model_name} to another name, it already exists in model_list")

        # 2. parse for argument parser
        parse_for_argparser(fn)

        # 3. parse for create_model
        parse_for_creator(fn, model_config, model_name)

    return fn


def parse_for_creator(fn, model_config, model_name):
    model_parameter = dict({k: v.default for k, v in signature(fn).parameters.items()})
    parameter, etc = model_config['parameter'], model_config['etc']

    for k, v in parameter.items():
        if k not in model_parameter:
            raise ValueError(f"{k} does not appear in {fn.__name__}, please update signature of model")
        model_parameter[k] = v

    assert _empty not in model_parameter, f"some required argument in {fn.__name__} does not in config"

    _name_to_model[model_name] = (fn, model_parameter, etc)


def parse_for_argparser(fn):
    model_argument = dict({k: v.annotation for k, v in signature(fn).parameters.items()})
    for name, type in model_argument.items():
        if type in [int, float, str]:
            parse_option = (type, 1)
        elif type in [Iterable[int], Iterable[float], Iterable[str]]:
            index = [Iterable[int], Iterable[float], Iterable[str]].index(type)
            val_type = [int, float, str][index]
            parse_option = (val_type, '+')
        else:
            parse_option = None

        if name in _argument_of_model and _argument_of_model[name] != parse_option:
            raise ValueError(f"{name} argument is duplicated but have different type")

        if parse_option:
            _argument_of_model[name] = parse_option


def create_model(model_name, **kwargs):
    # 1. load model and config
    creator, parameter, etc = _name_to_model.get(model_name, (None, None))

    if creator is None:
        raise ValueError(f"{model_name} is not found in list of models")

    # 2. update parameter by kwargs only if it appears in model config
    parameter = dict({k:kwargs.get(k, v) for k, v in parameter.items()})

    # Todo: Add pretrain handler

    return creator(**parameter)
