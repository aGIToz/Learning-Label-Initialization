import os
from tqdm import tqdm

opt = dict()

def generate_command(opt):
    cmd = 'python citeseer_.py'
    for key, val in opt.items():
        cmd += ' --' + key + ' ' + str(val)
    return cmd

def run(opt):
    os.system(generate_command(opt))

os.system('rm citeseer.txt')
for k in tqdm(range(100)):
    seed = k + 70 # k + default seed
    opt['rs'] = seed
    run(opt)
