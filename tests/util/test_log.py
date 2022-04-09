from box import Box

from pic.utils import Result


def test_log_save_results():
     args = Box(dict(output_dir='log', model_name='resnet50', dataset_type='CIFAR100', epoch=10, log_dir='log/resnet50_v1'))
     metric = dict(duration='00:00:10', best_acc=10.0, avg_top1_acc=11.0, avg_top5_acc=50.1)
     Result(args.output_dir).save_result(args, metric)
