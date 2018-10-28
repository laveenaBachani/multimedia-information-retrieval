import argparse
from Phase2.Modules import VisDescParser

def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--loc_id', help='Enter the location id -', type=str, required=True,
                                )
    argument_parse.add_argument('--model', help='The model you want to use(PCA/SVD/LDA)', type=str, required=True,
                                choices=['PCA', 'SVD', 'LDA'])
    argument_parse.add_argument('--k', help='The number semantics you want to find', type=int, required=True)
    args = argument_parse.parse_args()
    return args


# Main starts here

args = parse_args_process()
obj = VisDescParser()
obj.getTask4Items(args.loc_id, args.k,args.model)
# obj.getTask5Items("1",5,"LDA")
# out = obj.getTask5Items("LDA",2,"1","CM",5)
# print(out)
