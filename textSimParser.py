# Load libraries
import json
import math
import xml.etree.ElementTree as ET

class TextSimParser:

    @staticmethod
    def cosine_similarity(vector1, vector2, modelType):
        keys1 = set(vector1.keys())
        keys2 = set(vector2.keys())
        intersection_terms = keys1 & keys2
        v1_abs = 0
        v2_abs = 0
        dot_product_dict = {}
        for term in intersection_terms:
            dot_product_dict[term] = vector1[term][modelType]*vector2[term][modelType]
        dot_product = sum(dot_product_dict.values())
        sorted_list = [x for x in dot_product_dict.items()]
        sorted_list.sort(key=lambda x: x[1])  # sort by value
        sorted_list.reverse()
        for term in vector1.keys():
            v1_abs += vector1[term][modelType]*vector1[term][modelType]
        v1_abs = math.sqrt(v1_abs)
        for term in vector2.keys():
            v2_abs += vector2[term][modelType]*vector2[term][modelType]
        v2_abs = math.sqrt(v2_abs)
        if v1_abs == 0 or v2_abs == 0:
            cos_sim = 0
        else:
            cos_sim = float(dot_product)/(v1_abs*v2_abs)
        sim_dict = {'sim': cos_sim, 'major_contri': sorted_list[:3]}
        return sim_dict

    def get_k_similar_items(self, item_vectors, poi_item_id, modelType, k):
        sim = {}
        for item_id in item_vectors:
            if(item_id != poi_item_id):
                sim[item_id] = self.cosine_similarity(item_vectors[item_id], item_vectors[poi_item_id], modelType)
            #if int(item_id) in [135114844,288044415,132872354,135115501,135115317]:
            #    print(item_id, ' ', sim[item_id])
        sorted_list = [x for x in sim.items()]
        sorted_list.sort(key=lambda x: x[1]['sim'])  # sort by value
        sorted_list.reverse()
        #print(sorted_list[:k])
        return sorted_list[:k]

    @staticmethod
    def print_k_similar_items(k_similar_items, timeTaken):
        output = []
        for item in k_similar_items:
            major_contri = []
            for contrib in item[1]['major_contri']:
                major_contri.append(contrib[0])
            outputstr = str(item[0])+ " "+ str(item[1]['sim'])+ " "+ str(major_contri)
            output.append(outputstr)
        output.append("Time Taken :"+ str(timeTaken))
        return output

    @staticmethod
    def parse_txt_desc(filePath):
        newformat = {}
        with open(filePath) as openfileobject:
            for line in openfileobject:
                line = line.strip()
                # print(line)
                lineArr = line.split(' ')
                item_id = lineArr[0]
                termsInfo = {}
                i = 1
                while i < len(lineArr):
                    term = lineArr[i]
                    term = term.strip('"')
                    termdata = {}
                    termdata["TF"] = float(lineArr[i + 1])
                    termdata["DF"] = float(lineArr[i + 2])
                    termdata["TF-IDF"] = float(lineArr[i + 3])
                    termsInfo[term] = termdata
                    i += 4
                newformat[item_id] = termsInfo
            for item_id, termsInfo in newformat.items():
                for term, termdata in termsInfo.items():
                    termdata["TF"] = termdata["TF"] / len(termsInfo)
                    termdata["DF"] = termdata["DF"] / len(newformat)
                    termdata["TF-IDF"] = termdata["TF"]/termdata["DF"]
                    termsInfo[term] = termdata
                newformat[item_id] = termsInfo
        return newformat

    @staticmethod
    def get_topic_id_from_name(topics_data, topic_name):
        req_topic_id = -1
        for topic_id in topics_data:
            if topics_data[topic_id]['name'].strip() == topic_name.strip():
                req_topic_id = topic_id
        return req_topic_id

    def parse_txt_poi_desc(self, filePath, topics_data):
        newformat = {}
        with open(filePath) as openfileobject:
            for line in openfileobject:
                line = line.strip()
                lineArr = line.split(' ')
                poi_name = ''
                i = 0
                while(lineArr[i].find('"') == -1):
                    poi_name += ' ' + lineArr[i]
                    i += 1
                entity_id = self.get_topic_id_from_name(topics_data, poi_name)
                termsInfo = {}
                while i < len(lineArr):
                    term = lineArr[i]
                    term = term.strip('"')
                    termdata = {}
                    termdata["TF"] = float(lineArr[i + 1])
                    termdata["DF"] = float(lineArr[i + 2])
                    termdata["TF-IDF"] = float(lineArr[i + 3])
                    termsInfo[term] = termdata
                    i += 4
                newformat[entity_id] = termsInfo
        return newformat

    @staticmethod
    def parse_xml_topic(filePath):
        tree = ET.parse(filePath)
        root = tree.getroot()
        topics_data = {}
        for level1_child in root:
            topic_data = {}
            for level2_child in level1_child:
                if level2_child.tag == 'number':
                    topic_data[level2_child.tag] = level2_child.text
                elif level2_child.tag == 'latitude' or level2_child.tag == 'longitude':
                    topic_data[level2_child.tag] = float(level2_child.text)
                else:
                    topic_data[level2_child.tag] = level2_child.text
            topics_data[topic_data["number"]] = topic_data
        return topics_data

    @staticmethod
    def add_topic_name_in_topics(filePath, topics_data):
        with open(filePath) as openfileobject:
            for line in openfileobject:
                line = line.strip()
                line_arr = line.split("\t")
                location_name = line_arr[0]
                location_title = line_arr[1]
                for topic_id in topics_data:
                    if topics_data[topic_id]['title'] == location_title:
                        topics_data[topic_id]['name'] = location_name
        return topics_data