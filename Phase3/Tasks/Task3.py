import numpy as np
import argparse
import webbrowser
from Phase3.Modules import locationInfoParser

image_file = "../Data/devset_textTermsPerImage.txt"


def parse_args_process():
    # Argument Parser
    argument_parse = argparse.ArgumentParser()
    argument_parse.add_argument('--k', help='The number of most dominant images you want to find', type=int, required=True)
    args1 = argument_parse.parse_args()
    return args1


def page_rank(k, alph=0.15, diff=0.001):
    max_iter = 100
    # alph = 0.15

    adj_mtrx = np.load('adjMatrix_new.npy')
    row_sum = sum(adj_mtrx[0])
    adj_mtrx /= row_sum
    adj_mtrx = adj_mtrx.transpose()

    n = np.size(adj_mtrx[0])
    pr = (np.ones((n, 1)))/n
    pr1 = (np.ones((n, 1))) / n

    for i in range(max_iter):
        pr = alph*np.matmul(adj_mtrx, pr) + (1 - alph)*pr1

    find_k_most_relevant_images(pr, k)


def find_k_most_relevant_images(score, k):
    # This function finds the imageids of k images with largest pagerank value.
    # score is the the pagerank list.

    in_file = open(image_file, encoding="utf8")
    ids, id_score_dic = [], {}
    for line in in_file:
        ids.append(line.split()[0])

    ids.sort()
    i = 0
    for val in np.nditer(score):
        id_score_dic[ids[i]] = val
        i += 1

    image_id, pagerank = [], []
    sorted_by_score = sorted(id_score_dic.items(), key=lambda kv: kv[1])
    sorted_by_score.reverse()
    print("\nOutput - ")
    for i in range(k):
        print("id - "+sorted_by_score[i][0]+"\t score - "+str(sorted_by_score[i][1]))
        image_id.append(sorted_by_score[i][0])
        pagerank.append(sorted_by_score[i][1])

    visualization(image_id, pagerank)


def visualization(imageids, pageranks):
    # This function visualizes k images with largest pagerank value

    # finding location names
    locInfoParser = locationInfoParser.LocationInfoParser()
    locdict = locInfoParser.get_all_image_ids_locations()

    # f is the  task 3 html file
    file = "task3.html"
    f = open(file, 'w')

    args1 = parse_args_process()
    k = args1.k
    imgmessage = ""
    for id1 in range(len(imageids)):
        imgmessage += '<div class="column"> \n <img src="../img/' + str(imageids[id1]) + '.jpg"  title=" location : ' + locdict[imageids[id1]] + '\nPagerank : ' + str(pageranks[id1]) + ' " style="width:100%"> \n <p align="center"> Image Id:' + imageids[id1] + '</p> </div>'


    message = """
        <!DOCTYPE html>
        <html>
        <head>
        <style>
        * {
            box-sizing: border-box;
        }
        #main 
        {   
            border: 1px solid #c3c3c3;
            display: -webkit-flex;
            display: flex;
            -webkit-flex-wrap: wrap;
            flex-wrap: wrap;
            -webkit-align-content: center;
            align-content: center;
        }
        .column 
        {
            float: right;
            width: 16.66%;
            padding: 5px;           
        }
        /* Clearfix (clear floats) */
        .row::after 
        {
            content: "";
            clear: both;
            display: table;
        }
        @media screen and (max-width: 500px) 
        {
            .column 
            {
                width: 100%;
            }
        }
        </style>
        </head>
        <body>
        <h2 align="center">Task 3 Output</h2>
        <p align="center">""" + str(k) + """ most dominant images:</p>
        <div id="main" class="row"> """ + imgmessage + """
        </div>
        </body>
        </html>"""
    f.write(message)
    f.close()
    webbrowser.open_new_tab(file)


if __name__ == '__main__':
    args = parse_args_process()
    page_rank(args.k)
