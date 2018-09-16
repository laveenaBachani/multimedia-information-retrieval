import textSimParser
import visDescParser
import sys
import time
from pprint import pprint

basePath = "./"
devsetPath = basePath + "devset/"
textdescPath = devsetPath + "desctxt/"
visDesBasePath = devsetPath + "descvis/img/"
userWiseTextDescPath = textdescPath + "devset_textTermsPerUser.txt"
imageWiseTextDescPath = textdescPath + "devset_textTermsPerImage.txt"
locationWiseTextDescPath = textdescPath + "devset_textTermsPerPOI.txt"
xmlTopicsFilePath = devsetPath + "devset_topics.xml"
topicNamesFilePath = devsetPath + "poiNameCorrespondences.txt"
sampleInputPath = basePath + "actual_input.txt";
outputPath = basePath + "output.txt"


def readInputFromFile() -> dict:
    taskNo = -1
    taskTestCases = 0
    inputFileLineNo = 0
    taskCasesDict = {}
    with open(sampleInputPath) as openfileobject:
        for line in openfileobject:
            inputFileLineNo += 1
            line = line.strip()
            line = line.strip('# ')

            if not line:
                continue
            if taskTestCases != 0:
                taskCasesDict[taskNo].append(line)
                taskTestCases -= 1
                continue

            if line.find("SAMPLE") != -1 and line.find("INPUTS") != -1:
                continue
            if line.find("TASK") == -1 and line.find("TESTCASES") != -1:
                raise Exception("Expected Task at line: " + inputFileLineNo)

            lineArr = line.split(",")
            taskNo = lineArr[0].strip(" TASK")
            taskTestCases = lineArr[1].strip(" TESTCASES")
            if not taskNo.isnumeric() or not taskTestCases.isnumeric():
                raise Exception("Expected task number or test cases count not numeric at line: " + str(inputFileLineNo))
            taskNo = int(taskNo)
            taskTestCases = int(taskTestCases)
            taskCasesDict[taskNo] = []
    return taskCasesDict

def write_output(line):
    f = open(outputPath, "a+")
    f.write(line+"\n")
    f.close()
    print(line)

def clean_ouput():
    f = open(outputPath, "w+")
    f.close()

def parseAndPrintOutput(output,taskCasesDict):
    clean_ouput()
    for taskNo, testcases in output.items():
        write_output("### TASK " + str(taskNo) + " ###")
        for testcaseId, testcaseOutput in testcases.items():
            write_output("TESTCASE:" + str(testcaseId + 1) + " Input:" + str(taskCasesDict[taskNo][testcaseId]))
            for line in testcaseOutput:
                write_output(line)
            write_output("")
        write_output("")
    # pprint(output)

def executeTasks(taskCasesDict):
    print("Tasks read from input file")
    pprint(taskCasesDict)
    textParserObj = textSimParser.TextSimParser()
    topics_data = textParserObj.parse_xml_topic(xmlTopicsFilePath)
    topics_data = textParserObj.add_topic_name_in_topics(topicNamesFilePath, topics_data)

    output = {}
    parsedData = {}
    txtParsedData = {}
    for taskNo, testcases in taskCasesDict.items():
        print("Task No:", taskNo)
        output[taskNo] = {}
        if taskNo in [1,2,3]:
            if taskNo == 1:
                txtParsedData = textParserObj.parse_txt_desc(userWiseTextDescPath)
            elif taskNo == 2:
                txtParsedData = textParserObj.parse_txt_desc(imageWiseTextDescPath)
            else:
                txtParsedData = textParserObj.parse_txt_poi_desc(locationWiseTextDescPath, topics_data)
            for testcaseId, testcase in enumerate(testcases):
                startTime = time.time()
                input_arr = testcase.split(' ')
                poi_item_id = input_arr[0]
                model_type = input_arr[1]
                k = int(input_arr[2])
                k_similar = textParserObj.get_k_similar_items(txtParsedData, poi_item_id, model_type, k)
                timeTaken = time.time() - startTime
                output[taskNo][testcaseId] = textParserObj.print_k_similar_items(k_similar, timeTaken)
                pprint(output[taskNo][testcaseId])
        elif taskNo == 4:
            for testcaseId, testcase in enumerate(testcases):
                startTime = time.time()
                input_arr = testcase.split(' ')
                poi_item_id = input_arr[0]
                model_type = input_arr[1]
                k = int(input_arr[2])
                visDescParserObj = visDescParser.VisDescParser()
                parsedData = visDescParserObj.getAllLocationData(visDesBasePath, topics_data, model_type, parsedData)
                kSimilarItems = visDescParserObj.getKSimilarItems(parsedData, poi_item_id, model_type, k)
                timeTaken = time.time() - startTime
                output[taskNo][testcaseId] = visDescParserObj.printKSimilarItems(kSimilarItems, timeTaken)
                pprint(output[taskNo][testcaseId])
    
        elif taskNo == 5:
            for testcaseId, testcase in enumerate(testcases):
                startTime = time.time()
                input_arr = testcase.split(' ')
                poi_item_id = input_arr[0]
                k = int(input_arr[1])
                visDescParserObj = visDescParser.VisDescParser()
                parsedData = visDescParserObj.getAllLocationAllModelsData(visDesBasePath, topics_data, parsedData)
                kSimilarItems = visDescParserObj.getAllModelsKSimilarItems(parsedData, poi_item_id, k)
                timeTaken = time.time() - startTime
                output[taskNo][testcaseId] = visDescParserObj.printAllModelsKSimilarItems(kSimilarItems, timeTaken)
                pprint(output[taskNo][testcaseId])

    return output


taskCasesDict = readInputFromFile()
output = executeTasks(taskCasesDict)
parseAndPrintOutput(output, taskCasesDict)


'''
parser = textSimParser.TextSimParser()

topics_data = parser.parse_xml_topic(xmlTopicsFilePath)
topics_data = parser.add_topic_name_in_topics(topicNamesFilePath, topics_data)
newformat = parser.parse_txt_desc(imageWiseTextDescPath)
#newformat = parser.parse_txt_poi_desc(filePath, topics_data)

#f = open('demo2.json', 'a')
#f.write(json.dumps(newformat))


input_str = input()
input_arr = input_str.split(' ')
poi_item_id = input_arr[0]
model_type = input_arr[1]
k = int(input_arr[2])
k_similar = parser.get_k_similar_items(newformat, poi_item_id, model_type, k)
parser.print_k_similar_items(k_similar)


#print(topics_data)



#print(json.dumps(newformat))
'''