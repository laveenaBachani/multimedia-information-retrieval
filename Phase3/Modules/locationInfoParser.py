# Load libraries
import xml.etree.ElementTree as ET
class LocationInfoParser:

    RELATIVE_DEV_SET_PATH = "../Data/"

    XML_DIR = "xml/"

    TOPICS_INFO_FILE = "devset_topics.xml"

    def get_locations(self):
        topicsFilePath = self.RELATIVE_DEV_SET_PATH + self.TOPICS_INFO_FILE
        tree = ET.parse(topicsFilePath)
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

    def get_all_image_ids_locations(self):
        locations = self.get_locations()
        all_images_dict = {}
        for locationId, locationInfo in locations.items():
            locationTitle = locationInfo['title']
            location_xml_file = self.RELATIVE_DEV_SET_PATH + self.XML_DIR + locationTitle + '.xml'
            # print(location_xml_file)
            location_imgages_dict = self.get_location_imageIds(location_xml_file,locationTitle)
            all_images_dict = {**all_images_dict , **location_imgages_dict}
        return all_images_dict

    def get_location_imageIds(self, file_path, locationTitle):
        tree = ET.parse(file_path)
        root = tree.getroot()
        imageIds = {}
        for level1_child in root:
            imageId = level1_child.attrib['id']
            imageIds[imageId] = locationTitle
        return imageIds

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

if __name__ == '__main__':
    locInfoParser = LocationInfoParser()
    print(locInfoParser.get_all_image_ids_locations())