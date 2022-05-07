import os
import argparse
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--nowgts", action='store_true')
args = parser.parse_args()

opt = dict()

def generate_command_nowgts(opt):
    cmd = 'python pubmed_.py --nowgts'
    for key, val in opt.items():
        cmd += ' --' + key + ' ' + str(val)
    return cmd

def generate_command(opt):
    cmd = 'python pubmed_.py'
    for key, val in opt.items():
        cmd += ' --' + key + ' ' + str(val)
    return cmd

def run(opt, args):
    if args.nowgts:
        os.system(generate_command_nowgts(opt))
    else:
        os.system(generate_command(opt))

os.system('rm pubmed.txt')
for k in tqdm(range(100)):
    seed = k + 70 # k + default seed
    opt['rs'] = seed
    run(opt, args)
