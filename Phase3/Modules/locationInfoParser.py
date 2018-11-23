# Load libraries
import xml.etree.ElementTree as ET
class LocationInfoParser:

    RELATIVE_DEV_SET_PATH = "../Data/"

    XML_DIR = "xml/"

    TOPICS_INFO_FILE = "devset_topics.xml"

    TXT_PER_IMAGE_FILE = "devset_textTermsPerImage.txt"

    IMG_DIR = "img/"

    CLUSTER_ALGO_MAX_A_MIN_LABEL = "mam"

    CLUSTER_ALGO_SPECTRAL_PARTIONING_LABEL = "sp"

    CLUSTER_ALGO_MAX_A_MIN = "Max_A_Min"

    CLUSTER_ALGO_SPECTRAL_PARTIONING = "Spectral_Partitioning"

    SERVER_URL = "http://localhost:1337/"

    RELATIVE_TASKS_PATH = "../Tasks/"

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
        return

    def get_image_ids_of_node_ids(self):
        in_file = open(self.RELATIVE_DEV_SET_PATH + self.TXT_PER_IMAGE_FILE, encoding="utf8")
        ids = []
        for line in in_file:
            imageId = line.split()[0]
            ids.append(imageId)

        return ids

    def get_image_path(self, imageId, location):
        img_dir = self.IMG_DIR + location + "/"
        image_path = img_dir + imageId + ".jpg"
        return image_path

    def get_algo_label(self, cluster_algo):
        if cluster_algo == self.CLUSTER_ALGO_MAX_A_MIN:
            algo_label  = self.CLUSTER_ALGO_MAX_A_MIN_LABEL
        else:
            algo_label = self.CLUSTER_ALGO_SPECTRAL_PARTIONING_LABEL
        return algo_label

    def get_ouput_json_path(self, cluster_algo, taskId):
        algo_label = self.get_algo_label(cluster_algo)
        json_file_path = self.RELATIVE_DEV_SET_PATH + "task" + str(taskId) + "_" + algo_label + ".json"
        return json_file_path

    def get_ouput_html_path(self, cluster_algo, taskId):
        algo_label = self.get_algo_label(cluster_algo)
        html_file_path = self.SERVER_URL + "task" + str(taskId) + "_" + algo_label + ".html"
        return html_file_path

    def get_task_ouput_file_path(self, taskId):
        filePath = self.RELATIVE_TASKS_PATH + "task" + str(taskId) + "output.txt"
        return filePath


if __name__ == '__main__':
    locInfoParser = LocationInfoParser()
    print(locInfoParser.get_all_image_ids_locations())