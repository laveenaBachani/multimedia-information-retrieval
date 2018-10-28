import argparse
import numpy as np
from Phase2.Modules import VisDescParser as vd

def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--dim_red_algo', help='The model you want to use(PCA/SVD/LDA)', type=str, required=True,
                                choices=['PCA', 'SVD', 'LDA'])
    argument_parse.add_argument('--k', help='The number semantics you want to find', type=int, required=True)
    argument_parse.add_argument('--loc_id', help='Enter the location id -', type=str, required=True,
                                )
    argument_parse.add_argument('--vis_model', help='Enter the Visual descriptor model -', type=str, required=True,
                                )
    args = argument_parse.parse_args()
    return args

def clean_ouput():
    f = open("task4output.txt", "w+")
    f.close()

def write_output(line):
    f = open("task4output.txt", "a+")
    f.write(line+"\n")
    f.close()

def print_output(result , locInKSem, KSemInFet):
    write_output("Location in K semantics")
    for sem in locInKSem:
        write_output(np.array2string(sem))
    write_output("K semantics in features")
    for fet in KSemInFet:
        write_output(np.array2string(fet))
    write_output("Top K locations")
    for locId in result:
        write_output(str(locId))


# Main starts here
clean_ouput()
args = parse_args_process()
obj = vd.VisDescParser()
result , locInKSem, KSemInFet = obj.getTask4Items(args.dim_red_algo, args.k, args.loc_id, args.vis_model ,5)
print_output(result , locInKSem, KSemInFet)