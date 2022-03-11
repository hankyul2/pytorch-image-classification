from datetime import datetime


def print_metadata(model, train_dataset, test_dataset, args):
    title = 'INFORMATION'
    table = [('Project Name', args.project_name), ('Project Administrator', args.who),
             ('Experiment Name', args.exp_name), ('Experiment Start Time', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
             ('Experiment Model Name', args.model_name), ('Experiment Log Directory', args.log_dir)]
    print_tabular(title, table, args)

    title = "EXPERIMENT TARGET"
    table = [(target, str(getattr(args, target))) for target in args.exp_target]
    print_tabular(title, table, args)

    title = 'EXPERIMENT SETUP'
    table = [(target, str(getattr(args, target))) for target in [
        'train_resize', 'test_resize', 'crop_ptr', 'interpolation', 'mean', 'std',
        'hflip', 'auto_aug', 'cutmix', 'mixup', 'remode', 'aug_repeat',
        'model_name', 'ema', 'ema_decay', 'criterion', 'smoothing',
        'lr', 'epoch', 'optimizer', 'momentum', 'weight_decay', 'scheduler', 'warmup_epoch', 'batch_size'
    ]]
    print_tabular(title, table, args)

    title = 'DATA & MODEL'
    table = [('Model Parameters(M)', count_parameters(model)),
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