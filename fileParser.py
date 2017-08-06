import sys
import numpy as np

class FileParser:

    global attributes
    global data
    global data_matrix

    def parse_file(file_name):

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
            
                if decision == "yes\n":
                    substrings[len(substrings) - 1] = "yes"
                elif decision == "no\n":
                    substrings[len(substrings) - 1] = "no"
                    
                data.append(substrings)

            line_num += 1

        data_matrix = np.array(data)

        to_read.close()

        return data_matrix

    def get_attributes():
        global attributes
        return [att[0] for att in attributes]

    def get_attribute_values(att):
        global attributes

        index = [attr[0] for attr in attributes].index(att)
        return attributes[index][1]
        
    def get_datum(row_num):

        global data
        global data_matrix

        return data_matrix[row_num, :]

    def get_attribute(att):

        global attributes
        global data

        index = [attr[0] for attr in attributes].index(att)
        return [row[index] for row in data]

    def get_col(index):

        global data

        return [row[index] for row in data]

        parse_file(sys.argv[1])

        # attribute=[2,2,2,3,3]
        # vals=[2,3]
        # col=[0,0,1,1,0]
        # classes=[0,1]

#this is all a bit clumsy, I think it can be done much more neatly with data frames, maybe binding attribute/col into a frame and subsetting
#for example
class Tree:


    def __init__(self, root,children=None):
        self.value = root
        self.children = children

    def getRoot(self):
        return self.value

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

        # def toString(self):
        # if not self.isLeaf():
        # #print("got here")
        # for child in self.children:
        # child.toString() 
        # else:
        # print("LEAF")


class decisionTreeNode:
    
    def __init__(self,mat,classificationColIndex):
        self.data=mat
        self.classificationColumn = mat[:,classificationColIndex]
        self.entropy=self.computeEntropy(self.classificationColumn)
        self.best=self.bestSplit(mat,classificationColIndex)
        self.bestGain=self.best[0]
        self.bestAttribute=self.best[1]
        self.length=self.shape[1]


    def computeEntropy(self, column):
        possibleClasses=column.values()
        size=self.length
        H=0.0
        for value in possibleClasses:
            freq=column.count(value)
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

    def bestSplit(self, mat,classificationColIndex):
        bestAttribute=None
        gain=0
        for columnIndex in range(mat.shape[0]):
            curEntropy=attributeEntropy(mat,columnIndex,classificationColIndex)
            curGain=self.entropy-curEntropy
            if curGain>gain:
                gain=curGain
                bestAttribute=attribute

        return (gain, bestAttribute)

    def attributeEntropy(self, df,colIndex,classColIndex):
        attributeH=0.0
        size=self.length
        col=mat[:colIndex]
        vals=list(col.values())

        for value in vals:
            ratio=1.0*col.count(value)/size
            subset=mat[col==value,classColIndex]
            #print(subset)
            Hvalue=computeEntropy(subset)
            attributeH += Hvalue*ratio

        return attributeH




    #tree=Tree(0)
    #tree2=Tree(1)
    #tree3=Tree(2)
    #print(tree.getRoot())
    #tree.addChild(tree2)
    #tree2.addChild(Tree(3))
    #tree2.addChild(Tree(3))
    #tree.addChild(tree3)
    #tree.toString()
    #print(tree.isLeaf())
    #print(tree2.getChildren)
    
    DTN = decisionTreeNode(FileParser.parse_file(sys.argv[1]), 1)
    
    
    # Code to test functionality of file parsing

    #print("\n\n")
    #print("Attributes: " + str(get_attributes()))
    #print("\n\n")
    #print("Values of temperature: " + str(get_attribute_values("temperature")))
    #print("\n\n")
    #print("3rd datum: " + str(get_datum(2)))
    #print("\n\n")
    #print("Temperature Att: " + str(get_attribute("temperature")))
    #print("\n\n")
    #print("3rd Column: " + str(get_col(2)))
