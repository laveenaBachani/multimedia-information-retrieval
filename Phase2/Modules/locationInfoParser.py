# Load libraries
import xml.etree.ElementTree as ET
class LocationInfoParser:

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
                topic_data['name'] = ''
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