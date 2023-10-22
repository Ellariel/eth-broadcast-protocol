import os, argparse

parser = argparse.ArgumentParser()
parser.add_argument('--type', default='ethereum', type=str)
parser.add_argument('--duration', default=60, type=int, nargs="+")
parser.add_argument('--k', default=1000, type=int)
parser.add_argument('--test', default='PPO-env-1.sav.0.442', type=str)

parser.add_argument('--log', default='#', type=str)
parser.add_argument('--output', default='#', type=str)
parser.add_argument('--error', default='', type=str)

args = parser.parse_args()
args.duration = set(args.duration)
print(f"type:{args.type}, k:{args.k}, duration:{args.duration}, test:{args.test}")

for d in args.duration:
    print(f"duration:{d}")
    line = f'''executable = /bin/bash 
arguments = -i conda_activate.sh blocksim exp.py --type {args.type} --k {args.k} --duration {d} --test {args.test}
transfer_input_files = conda_activate.sh, exp.py 
{args.log}log = ./logs/exp.log 
{args.output}output = ./logs/exp_stdout.txt
{args.error}error = ./logs/exp_stderr.txt
should_transfer_files = IF_NEEDED 
request_gpus = 1
queue
'''
    with open(f'./run_exp.sub', 'wt') as file:
        file.writelines(line)
    os.system(f'condor_submit ./run_exp.sub')
    os.remove('./run_exp.sub')