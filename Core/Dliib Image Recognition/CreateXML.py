import glob

from yattag import Doc, indent
import numpy as ny


class XmlDataCreator:
    fileName = ""
    valuesList = []
    x = 0
    y = 0
    # def __init__(self):
    #


doc, tag, text = Doc().tagtext()
pictureDataFiles = glob.glob("annotation/*")
#
# with tag('root'):
#     with tag('doc'):
#         with tag('part', name='blah'):
#             text('some value1')
#         with tag('part', name='asdfasd'):
#             text('some value2')
#
# result = indent(
#     doc.getvalue(),
#     indentation=' ' * 4,
#     newline='\r\n'
# )
#
# print(result)
print("Amount of files: \n" + str(len(pictureDataFiles)))

outputXmlFile = ""
singleModule = XmlDataCreator
counter = 0

for dataFile in pictureDataFiles:
    print(dataFile)
    counter += 1
    print(counter)
    with open(dataFile) as openFile:
        content = openFile.readlines()
        # print(content[0])

        # pobranie nazwy pliku graficznego
        singleModule.fileName = content[0]
        print(singleModule.fileName)
        loopcounter = 0
        for i in range(1, len(content) - 1):
            # print(loopcounter)
            # print(content[i])
            strA = content[i].rstrip().partition(" , ")
            singleModule.valuesList.append([float(strA[0]), float(strA[2])])
            #
            # print(strA)
            loopcounter += 1

        print(singleModule.valuesList)
        singleModule.valuesList = []
