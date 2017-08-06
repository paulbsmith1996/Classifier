import sys
import numpy as np

class FileParser:

    global attributes
    global data
    global data_matrix

    def parse_file(self,file_name):

        global attributes
        global data_matrix
        global data

        to_read = open(file_name)

        relation_tag = "@relation"
        att_tag = "@attribute"
        data_tag = "@data"

        attributes = []
        data = []

        data_start = sys.maxint
        line_num = 0

        for line in to_read:


            fixed_line = line.lower()

            if fixed_line.startswith(relation_tag, 0, len(relation_tag)):
                pass
            elif fixed_line.startswith(att_tag, 0, len(att_tag)):

                fw_index = fixed_line.index(" ", len(data_tag), len(fixed_line)) + 1
                sw_index = fixed_line.index(" ", fw_index + 1, len(fixed_line))

                attribute_values = []

                av_index = fixed_line.find("{")
                av_end = fixed_line.find("}")

                if av_index < 0:
                    av_start = sw_index + 1
                    attribute_values.append(fixed_line[av_start: len(fixed_line) - 1])
                elif av_end > 0:
                    av_start = av_index + 1
                    av_end = av_end


                    values = fixed_line[av_start: av_end].split(",")

                    attribute_values = [value.strip(" ") for value in values]

                attributes.append((fixed_line[fw_index: sw_index], attribute_values))
            elif fixed_line.startswith(data_tag, 0, len(data_tag)):
                data_start = line_num
                pass
            elif line_num > data_start:
                # This is a datum                                                                             


                fixed_line.rstrip("\n")
                substrings = fixed_line.split(",")

                for strng in substrings:
                    strng.strip()

                decision = substrings[len(substrings) - 1]

                if decision.endswith("\n"):
                    substrings[len(substrings) - 1] = decision.strip("\n")


                if not line.startswith("%"):
                    data.append(substrings)

            line_num += 1

        data_matrix = np.array(data)

        to_read.close()
        return (data_matrix, attributes)

#this is all a bit clumsy, I think it can be done much more neatly with data frames, maybe binding attribute/
#col into a frame and subsetting  
#for example                                                                                                  
class Tree:


    def __init__(self, value,children=None):
        self.value = value
        self.children = children

    def getRoot(self):
        return self

    def getChildren(self):
        return self.children

    def isLeaf(self):
        return self.children==None

    def getLeaves(self):
        if self.isLeaf():
            return self.value
        else:
            return [child.getLeaves() for child in self.children]

    def addChild(self,child):
        if not self.children:
            self.children=[child]
        else:
            self.children.append(child)

    def removeChild(self,index):
        del self.children[index]

    def to_string(self):
        return self.to_string_helper(self.getRoot(), 1)

    def to_string_helper(self, tree, indent):

        TAB_LENGTH = 4
        rep = str(tree.value)

        if not tree.isLeaf():

            for child in tree.getChildren():

                rep += "\n"

                i = 0
                while i < (TAB_LENGTH * indent):
                    rep += " "
                    i += 1


                rep += str(tree.to_string_helper(child, indent + 1))
                tree.to_string_helper(child, indent + 1)


        return rep



class decisionTreeNode:

    def __init__(self,mat, attrs, classificationColIndex):
        self.data=mat
        self.length = mat.shape[0]
        self.attributes = attrs
        self.classificationColumn = mat[:, classificationColIndex]
        self.entropy=self.computeEntropy(classificationColIndex)
        self.best=self.bestSplit(classificationColIndex)
        self.bestGain=self.best[0]
        self.bestAttribute=self.best[1]

        #print(mat.shape)                                                                                     


    def computeEntropy(self, col_index,column=None):
        if column is None:
            column=list(self.data[:,col_index])

        possibleClasses = self.attributes[col_index][1]

        size=self.length
        H=0.0
        for value in possibleClasses:
            freq = list(column).count(value)
            if freq>0:
                ratio=1.0*freq/size
                H-=ratio*np.log2(ratio)

        return H

    def isPure(self):
        return self.entropy==0

    def split(self, attributeIndex): #WE NEED TO CHANGE THIS TO HANDLE CONTINUOUS                             
        attribute=self.data[:,attributeIndex]
        values=attribute.values()
        newNodes=[]
        for value in values:
            newNode=decisionTreeNode(self.data[attribute==value,:],classificationColName)
            newNodes.append(newNode)
        return newNodes

    def bestSplit(self, classificationColIndex):
        bestAttribute=None
        gain=0

        for columnIndex in range(self.data.shape[1] - 1):
            if not columnIndex==classificationColIndex:
                curEntropy=self.attributeEntropy(columnIndex, classificationColIndex)
                curGain=self.entropy-curEntropy
                if curGain>gain:
                    gain=curGain
                    bestAttribute=attributes[columnIndex][0]

        return (gain, bestAttribute)

    def attributeEntropy(self, colIndex, classColIndex):
        attributeH=0.0
        size=self.length
        col = list(self.data[:, colIndex])
        vals = list(attributes[colIndex][1])

        #print("size: " + str(size))                                                                          
        #print("\ncol: " + str(col) + "\n\n")                                                                 

        for value in vals:

            #print(col.count(value))                                                                          
            count = col.count(value)

            if count > 0:
                ratio=1.0*col.count(value)/size

                vect = [row[colIndex]==value for row in self.data]
                subset=self.data[vect,classColIndex]
                Hvalue = self.computeEntropy(classColIndex, subset)
                attributeH += Hvalue*ratio


        return attributeH




#fp = FileParser()
#tp = fp.parse_file(sys.argv[1])
#DTN = decisionTreeNode(tp[0], tp[1], len(tp[1]) - 1)
#print("best gain: " + str(DTN.bestGain))
#print("best attribute: " + str(DTN.bestAttribute))

tree=Tree(0)
tree2=Tree(1)
tree3=Tree(2)
print("\n\nroot: "+ str(tree.getRoot()) + "\n")
print("root value: " + str(tree.value) + "\n")
tree2.addChild(Tree(3))
tree2.addChild(Tree(3))
tree.addChild(tree2)
tree.addChild(tree3)
tree3.addChild(Tree(4))
tree3.addChild(Tree(0))
print("tree to string:\n\n" + str(tree.to_string()) + "\n")
print("is tree a leaf?: " + str(tree.isLeaf()) + "\n")
print("tree2 children: " + str(tree2.getChildren) + "\n")
