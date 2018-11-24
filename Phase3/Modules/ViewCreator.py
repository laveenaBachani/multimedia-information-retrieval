import json
import os, webbrowser
from Phase3.Modules import locationInfoParser

class Visualizer:

    def __init__(self, taskId):
        self.taskId = taskId
        self.locInfoParser = locationInfoParser.LocationInfoParser()

    def visualize_clusters(self, clusters, cluster_algo):
        locInfoParser = self.locInfoParser
        allNodeIdsImageId = locInfoParser.get_image_ids_of_node_ids()
        allImageIdsLocation = locInfoParser.get_all_image_ids_locations()
        clusters_dict = {}
        for lable,cluster in clusters.items():
            clusters_dict[lable] =[]
            for nodeId in cluster:
                imageId = allNodeIdsImageId[nodeId]
                location = allImageIdsLocation[imageId]
                image_path = locInfoParser.get_image_path(imageId,location)
                clusters_dict[lable].append(image_path)
        json_output_path = locInfoParser.get_ouput_json_path(cluster_algo, self.taskId)
        json_output = open(json_output_path, 'w')
        json_output.write(json.dumps(clusters_dict))
        json_output.close()
        #print(json_output_path)
        #print("Cluster Algo:",cluster_algo)
        #print(clusters_dict)
        self.write_output("Cluster Algo:"+str(cluster_algo))
        self.write_output(json.dumps(clusters_dict))
        html_output_file = locInfoParser.get_ouput_html_path(cluster_algo, self.taskId)
        webbrowser.open_new_tab(html_output_file)

    def clean_ouput(self):
        output_file_path = self.locInfoParser.get_task_ouput_file_path(self.taskId)
        f = open(output_file_path, "w+")
        f.close()

    def write_output(self, line):
        output_file_path = self.locInfoParser.get_task_ouput_file_path(self.taskId)
        f = open(output_file_path, "a+")
        f.write(line + "\n")
        print(line)
        f.close()