from datetime import datetime

try:
    from deepspeed.profiling.flops_profiler import get_model_profile
    deepspeed = True
except:
    deepspeed = False


def print_metadata(model, train_dataset, test_dataset, args):
    title = 'INFORMATION'
    table = [('Project Name', args.project_name), ('Project Administrator', args.who),
             ('Experiment Name', args.exp_name), ('Experiment Start Time', args.start_time),
             ('Experiment Model Name', args.model_name), ('Experiment Log Directory', args.log_dir)]
    print_tabular(title, table, args)

    title = "EXPERIMENT TARGET"
    table = [(target, str(getattr(args, target))) for target in args.exp_target]
    print_tabular(title, table, args)

    title = 'EXPERIMENT SETUP'
    table = [(target, str(getattr(args, target))) for target in [
        'train_size', 'train_resize_mode', 'random_crop_pad', 'random_crop_scale', 'random_crop_ratio', 'test_size', 'test_resize_mode', 'center_crop_ptr',
        'interpolation', 'mean', 'std', 'hflip', 'auto_aug', 'cutmix', 'mixup', 'remode', 'aug_repeat',
        'model_name', 'ema', 'ema_decay', 'criterion', 'smoothing',
        'lr', 'epoch', 'optimizer', 'momentum', 'weight_decay', 'scheduler', 'warmup_epoch', 'batch_size'
    ]]
    print_tabular(title, table, args)

    if deepspeed and args.print_flops:
        flops = get_model_profile(model, input_res=(1, 3, 224, 224), print_profile=False, detailed=False)[0]
    else:
        flops = 'install deepspeed & enable print-flops'

    title = 'DATA & MODEL'
    table = [('Model Parameters(M)', count_parameters(model)),
             (f'Model FLOPs(3, 224, 224)', flops),
             ('Number of Train Examples', len(train_dataset)),
             ('Number of Valid Examples', len(test_dataset)),
             ('Number of Class', args.num_classes),]
    print_tabular(title, table, args)

    title = 'TERMINOLOGY'
    table = [('Batch', 'Time for 1 epoch in seconds'), ('Data', 'Time for loading data in seconds'),
             ('F+B+O', 'Time for Forward-Backward-Optimizer in seconds'), ('Top-1', 'Top-1 Accuracy'),
             ('Top-5', 'Top-5 Accuracy')]
    print_tabular(title, table, args)

    args.log("-" * 81)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_tabular(title, table, args):
    title_space = int((81 - len(title)) / 2)
    args.log("-" * 81)
    args.log(" " * title_space + title)
    args.log("-" * 81)
    for (key, value) in table:
        args.log(f"{key:<25} | {value}")