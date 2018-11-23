import argparse
import os,webbrowser
from Phase3.APIs.generic_apis import *
import numpy as np
np.set_printoptions(threshold=np.nan)
import matplotlib.pyplot as plt

image_file = "../../Phase2/Data/devset_textTermsPerImage.txt"

def visualization(imageids):
    file="task4imageoutput.html"
    f = open(file,'w')
    imgmessage = ""

    for id1 in imageids:
        imgmessage += '<div class="column"> \n <img src="C:\\Users\\Shubham\\Downloads\\img1\\' + id1 + '.jpg"  title="' + id1 +' " style="width:100%"> \n <p align="center"> Image Id:'+id1+'</p> </div>'

    message = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        * {
            box-sizing: border-box;
        }
        #main {
           
            border: 1px solid #c3c3c3;
            display: -webkit-flex;
            display: flex;
            -webkit-flex-wrap: wrap;
            flex-wrap: wrap;
            -webkit-align-content: center;
            align-content: center;
        }
        
        .column {
            float: right;
            width: 16.66%;
            padding: 5px;
            
        }
        
        /* Clearfix (clear floats) */
        .row::after {
            content: "";
            clear: both;
            display: table;
        }
    
        @media screen and (max-width: 500px) {
            .column {
                width: 100%;
            }
        }
        </style>
        </head>
        <body>
        <h2 align="center">Task 4 Output</h2>
        <p align="center">K Top most relavant images:</p>
        <div id="main" class="row"> """ + imgmessage +"""
        </div>
        </body>
        </html>"""
    f.write(message)
    f.close()
    webbrowser.open_new_tab(file)

def imageids(id1,id2,id3):
    in_file = open(image_file, encoding="utf8")
    ids = []
    pos=[]
    for line in in_file:
        ids.append(line.split()[0])
    ids.sort();
    for id in range(0,len(ids)):
        if ids[id]==id1:
            pos.append(id)
        elif ids[id]==id2:
            pos.append(id)
        elif ids[id]==id3:
            pos.append(id)
    return pos

# Argument Parser
def parse_args_process():
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The no of relevant images', type=int, required=True)
    argument_parse.add_argument('--id1', help='image id 1', type=str, required=True)
    argument_parse.add_argument('--id2', help='image id 2', type=str, required=True)
    argument_parse.add_argument('--id3', help='image id 3', type=str, required=True)
    args = argument_parse.parse_args()
    return args

def find_k_most_relevant_images(score, k):
    in_file = open(image_file, encoding="utf8")
    ids = []
    id_score_dic = {}
    for line in in_file:
        ids.append(line.split()[0])

    i = 0
    for val in np.nditer(score):
        id_score_dic[ids[i]] = val
        i += 1

    sorted_by_score = sorted(id_score_dic.items(), key=lambda kv: kv[1])
    sorted_by_score.reverse()
    imageids=[]
    pageranks=[]
    for i in range(k):
        imageids.append(sorted_by_score[i][0])
        pageranks.append(sorted_by_score[i][1])
        print("id - " + sorted_by_score[i][0] + "\t score - " + str(sorted_by_score[i][1]))
    #plotraph(imageids,pageranks)
    visualization(imageids)


def ppr():
    alpha=.65
    args = parse_args_process()
    k=args.k
    id1=args.id1
    id2=args.id2
    id3=args.id3
    adj_mtrx=np.load('adjMatrix_new.npy')
    row_sum = sum(adj_mtrx[0])
    adj_mtrx /= row_sum
    adj_mtrx = adj_mtrx.transpose()
    n = np.size(adj_mtrx[0])
    pr = (np.zeros((n, 1))) 
    pos=imageids(id1,id2,id3)
    #print(pos[0],pos[1],pos[2])
    for i in range(n):
        if(i in pos):
            pr[i][0]=1/3

    for i in range(100):
        pr = alpha*np.matmul(adj_mtrx, pr) + (1 - alpha)*pr

    #print(-np.sort(-pr,axis=0))
    find_k_most_relevant_images(pr, k)



if __name__ == '__main__':
    ppr()
