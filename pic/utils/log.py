import csv
import json
import os
from pathlib import Path

from filelock import FileLock


def csv2dict(csv_data):
    return {col: [row[c] for row in csv_data[1:]] for c, col in enumerate(csv_data[0])}


def write_csv(result_path, data, mode='w'):
    with open(result_path, mode, newline='') as f:
        writer = csv.writer(f)
        for row in data:
            writer.writerow(row)

class Result:
    def __init__(self, log_path='log'):
        self.result_path = os.path.join(log_path, 'result.csv')

        # Todo: Change This whenever you apply this utils to other domain
        self.headers = ['no', 'setup', 'model_name', 'avg_top1_acc', 'avg_top5_acc', 'duration', 'log_dir',
                        'start_time', 'dataset_type',  'epoch', 'best_acc']

        self.setup_directory()
        self.setup_logfile()

    def setup_directory(self):
        Path(os.path.dirname(self.result_path)).mkdir(exist_ok=True, parents=True)

    def setup_logfile(self):
        if not Path(self.result_path).exists():
            write_csv(self.result_path, [self.headers], mode='w')

    def read_result(self):
        csv_data = []
        with open(self.result_path, 'r') as f:
            for line in csv.reader(f):
                csv_data.append(line)
        return csv_data, csv2dict(csv_data)

    def arg2result(self, args, metric):
        result = [self.get_no()]
        for column_name in self.headers[1:]:
            if hasattr(args, column_name):
                result.append(getattr(args, column_name))
            elif metric.get(column_name, None):
                result.append(metric.get(column_name, None))
            else:
                result.append('')
                print("Args and Metric object does not have : {}".format(column_name))
        return result

    def get_no(self):
        csv_list, csv_dict = self.read_result()
        return len(csv_list)

    def summary(self, args, metric):
        space = 16
        num_metric = 4
        duration, best_acc, top1, top5 = list(metric.values())
        args.log('-' * space * num_metric)
        args.log(("{:>16}" * num_metric).format('Duration', 'Best Top-1 Acc', 'Avg Top-1 Acc', 'Avg Top-5 Acc'))
        args.log('-' * space * num_metric)
        args.log(f"{duration:>{space}}{best_acc:{space}.4f}{top1:{space}.4f}{top5:{space}.4f}")
        args.log('-' * space * num_metric)

        with FileLock("{}.lock".format(self.result_path)):
            result = self.arg2result(args, metric)
            write_csv(self.result_path, [result], mode='a')

    def dump_args(self, args):
        with open(os.path.join(args.log_dir, 'cmd.json'), 'wt') as f:
            keys_to_remove = ('device', 'logger', 'log')
            cmd = dict({key: val for key, val in args.__dict__.items() if key not in keys_to_remove})
            json.dump(cmd, f, indent=4, ensure_ascii=False)

    def dump_metric(self, log_dir, top1_list, top5_list):
        metric = [['top1', 'top5']]
        metric.extend([[round(float(top1), 4), round(float(top5), 4)] for top1, top5 in zip(top1_list, top5_list)])
        write_csv(os.path.join(log_dir, 'metric.csv'), metric)

    def save_result(self, args, top1_list, top5_list, metric):
        self.summary(args, metric)
        self.dump_args(args)
        self.dump_metric(args.log_dir, top1_list, top5_list)
