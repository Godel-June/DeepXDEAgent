import argparse
import os.path as osp

class ConfigOptions():
    def initialize(self):
        parser = argparse.ArgumentParser(description="DeepXDE Agent")
        # parser.add_argument('--dataset', type=str, default='medqa')
        # parser.add_argument('--model', type=str, default='gpt-4o-mini')
        parser.add_argument('--model', type=str, default='deepseek-chat', help="The model to use for the agent.")
        parser.add_argument('--difficulty', type=str, default='adaptive', help="The difficulty level of the user's query.")
        # parser.add_argument('--num_samples', type=int, default=100)
        # parser.add_argument('--num_samples', type=int, default=4)

        # parser.add_argument()

        return parser.parse_args()

    def print_options(self, args):
        message = ''
        message += '----------------- Options ---------------\n'
        for k, v in sorted(vars(args).items()):
            comment = ''
            # {:>25} 表示 25 个字符，右对齐
            message += '{:>25}: {:<30}{}\n'.format(str(k), str(v), comment)
        message += '----------------- End -------------------'
        print(message)

        # save to file
        file_name = osp.join(args.snapshot_dir, 'log.txt')
        with open(file_name, 'wt') as args_file:
            args_file.write(message)
            args_file.write('\n')
