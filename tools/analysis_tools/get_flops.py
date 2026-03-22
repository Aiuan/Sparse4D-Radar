# Copyright (c) OpenMMLab. All rights reserved.
import argparse
import os
from functools import partial
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel
from mmdet.models import build_detector
from mmdet.datasets import build_dataset, build_dataloader

# try:
#     from mmcv.cnn import get_model_complexity_info
# except ImportError:
#     raise ImportError('Please upgrade mmcv to >0.6.2')

# support new modules in FLOPs
from flops_counter import get_model_complexity_info


def parse_args():
    parser = argparse.ArgumentParser(description='Train a detector')
    parser.add_argument('config', help='train config file path')
    parser.add_argument(
        '--shape',
        type=int,
        nargs='+',
        default=[40000, 4],
        help='input point cloud size')
    parser.add_argument(
        '--modality',
        type=str,
        default='point',
        choices=['point', 'image', 'multi'],
        help='input data modality')
    parser.add_argument(
        '--cfg-options',
        nargs='+',
        action=DictAction,
        help='override some settings in the used config, the key-value pair '
             'in xxx=yyy format will be merged into config file. If the value to '
             'be overwritten is a list, it should be like key="[a,b]" or key=a,b '
             'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
             'Note that the quotation marks are necessary and that no white space '
             'is allowed.')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    if args.modality == 'point':
        assert len(args.shape) == 2, 'invalid input shape'
        input_shape = tuple(args.shape)
    elif args.modality == 'image':
        if len(args.shape) == 1:
            input_shape = (3, args.shape[0], args.shape[0])
        elif len(args.shape) == 2:
            input_shape = (3,) + tuple(args.shape)
        else:
            raise ValueError('invalid input shape')
    elif args.modality == 'multi':
        raise NotImplementedError(
            'FLOPs counter is currently not supported for models with '
            'multi-modality input')

    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    # import modules from string list.
    if cfg.get("custom_imports", None):
        from mmcv.utils import import_modules_from_strings

        import_modules_from_strings(**cfg["custom_imports"])

    # import modules from plguin/xx, registry will be updated
    if hasattr(cfg, "plugin"):
        if cfg.plugin:
            import importlib

            if hasattr(cfg, "plugin_dir"):
                plugin_dir = cfg.plugin_dir
                _module_dir = os.path.dirname(plugin_dir)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]

                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)
            else:
                # import dir is the dirpath for the config file
                _module_dir = os.path.dirname(args.config)
                _module_dir = _module_dir.split("/")
                _module_path = _module_dir[0]
                for m in _module_dir[1:]:
                    _module_path = _module_path + "." + m
                print(_module_path)
                plg_lib = importlib.import_module(_module_path)

    # build the model and load checkpoint
    model = build_detector(
        cfg.model,
        train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    model = MMDataParallel(model, device_ids=[0])
    model.eval()

    # select input_constructor according to cfg
    construct_input = partial(construct_input_from_cfg, dataset_cfg=cfg.data.test)

    flops, params = get_model_complexity_info(model, input_shape, input_constructor=construct_input)
    split_line = '=' * 30
    print(f'{split_line}\nInput shape: {input_shape}\n'
          f'Flops: {flops}\nParams: {params}\n{split_line}')
    print('!!!Please be cautious if you use the results in papers. '
          'You may need to check if all ops are supported and verify that the '
          'flops computation is correct.')


def construct_input_from_cfg(input_shape, dataset_cfg):
    dataset = build_dataset(dataset_cfg)

    data_loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=0,
        dist=False,
        shuffle=False,
    )

    return next(iter(data_loader))


if __name__ == '__main__':
    main()
