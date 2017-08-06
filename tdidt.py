#
# Authors: Paul Baird-Smith and david Burt
# CS373: Artificial Intelligence,
# Project 5: Classifier Learning
# April 18, 2017
#
# This classifier takes the name of a .arff file as input,
# parses it, and then creates a classification tree given the data.
#

import sys
import numpy as np
import util

# Parses a .arff file and ultimately will yield a tuple of (attributes, data)
#
# attributes - holds the names and values of the attributes we are classifying over
# data - holds all the data in an np.array, with all values in lower case and with no
# white spaces

global data

class FileParser:

    #
    # Parses file and returns a tuple tp such that
    # tp[0] is an np.array of the data in the .arff file
    # tp[1] is a list of tuples of attributes and possible values the attr can take
    #
    # file_name is the name of the .arff file to be parsed
    #
    def parse_file(self, file_name):

        # Open .arff file
        to_read = open(file_name)

        # Assign variables to the different line prefixes of a .arff file
        relation_tag = "@relation"
        att_tag = "@attribute"
        data_tag = "@data"

        # Initialize empty lists for attributes and data
        attributes = []
        data = []

        # Until we read a data_tag, we do not want to read any data
        # data_start represents the line number at which the block of data begins
        data_start = sys.maxint

        # We will increment line_num as we iterate through the lines of the file
        line_num = 0

        # Iterate through lines in opened file
        for line in to_read:

            # Put the current line to be all lower-case. This simplifies String and tag matching
            fixed_line = line.lower()

            # Current line's prefix is the relation tag
            if fixed_line.startswith(relation_tag, 0, len(relation_tag)):
                # Don't do anything with the relation tag, though in the future,
                # this tag might have to be recognized.
                pass

            # Current line's prefix is the attribute tag
            elif fixed_line.startswith(att_tag, 0, len(att_tag)):

                # fw_index is the index of the first white space in fixed_line after the tag
                fw_index = fixed_line.index(" ", len(data_tag), len(fixed_line)) + 1

                # The index of the second white space in fixed_line after the tag
                sw_index = fixed_line.index(" ", fw_index + 1, len(fixed_line))

                # List holding the possible values of the current attribute
                attribute_values = []

                # Indices representing the start and end of the list of values,
                # if there is such a list
                av_index = fixed_line.find("{")
                av_end = fixed_line.find("}")

                # av_index < 0 implies that the attribute is continous-valued.
                # This means that we are looking for an value label instead of 
                # an value list.
                if av_index < 0:

                    # Start of the value label is the character after the second white space
                    av_start = sw_index + 1

                    # For a continuous-valued attr, we create a list of only its value label 
                    attribute_values.append(fixed_line[av_start: len(fixed_line) - 1])

                # Check that the line actually closes its attr value list
                elif av_end > 0:
                    # Our list starts at the character after "{"
                    av_start = av_index + 1

                    # Values is the list of substrings obtained by splitting our value list
                    # at each ","
                    values = fixed_line[av_start: av_end].split(",")

                    # Remove leading and trailing white spaces from all values
                    attribute_values = [value.strip(" ") for value in values]

                # Update our attrtibutes so that we have the new attribute and its possible
                # values
                attributes.append((fixed_line[fw_index: sw_index], attribute_values))

            # Current line's prefix is the data tag
            elif fixed_line.startswith(data_tag, 0, len(data_tag)):
                # Update the index of the first line of the data block
                data_start = line_num
                pass
            
            # Current line appears after the data tag. It is a datum unless prefixed by "%"
            elif line_num > data_start:

                # Remove the new line character at the end of the line
                fixed_line.rstrip("\n")
                
                # Get the values of each attribute
                substrings = fixed_line.split(",")

                # Remove leading and trailing white spaces for each attribute value
                for strng in substrings:
                    strng.strip()

                # This is the final value of the datum and will end with a new line character
                decision = substrings[len(substrings) - 1]

                # Remove the new line character from the final value
                if decision.endswith("\n"):
                    substrings[len(substrings) - 1] = decision.strip("\n")

                # Line is a comment and should not be considered as a datum
                if not line.startswith("%"):
                    data.append(substrings)

            # Increment line_num on every pass
            line_num += 1

        # Wrap our data into and np.array so that we inherit its functionality
        data_matrix = np.array(data)

        # Close file
        to_read.close()

        # Return tuple of np.array of data and a list of tuples of attributes and their values
        return (data_matrix, attributes)



class Tree:


    def __init__(self, value,children=None, parent=None):
        self.value = value
        self.children = children
        self.parent = parent

    def getRoot(self):
        return self

    def getParent(self):
        return self.parent

    def getChildren(self):
        return self.children

    def isLeaf(self):
        return self.children==None

    def getLeaves(self):
        if self.isLeaf():
            return self.value
        else:
            return [child.getLeaves() for child in self.children]

    def addChild(self, child):
        child.parent = self
        if not self.children:
            self.children=[child]
        else:
            self.children.append(child)
            
    def addLeaf(self, leaf, parent):
        if self.value==parent:
            self.addChild(Tree(leaf))
        else:
            if not self.isLeaf():
                for child in self.getChildren():
                    child.addLeaf(leaf,parent)

    def removeChild(self,index):
        del self.children[index]

    def to_string(self):
        return self.to_string_helper(self.getRoot(), 1)

    def to_string_helper(self, tree, indent):

        TAB_LENGTH = 4
        rep = str(tree.value.bestAttribute)

        if rep == "None":
            rep = tree.value.majorityVote()

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



class decisionTree:

    def __init__(self, mat, attrsAndValues, classificationColIndex, weights=None, last =None,test=False):
        self.data = mat
        self.lastSplit = last
        self.length = mat.shape[0]
        self.ccIndex = classificationColIndex

        #weights default to be uniform
        if weights is None:
            weights=[1]*self.length

        self.weights = weights

        self.attrsAndValues = attrsAndValues
        self.attributes = [atv[0] for atv in attrsAndValues]
        self.attValues = [atv[1] for atv in attrsAndValues]
        self.classificationColumn = mat[:, classificationColIndex]

        self.entropy = self.computeEntropy()
        if not test:
            self.best = self.bestSplit()
            self.bestGain = self.best[0]
            self.bestAttribute = self.best[1]

        

    #computes the most common class (according to the weights) of a given tree node
    def majorityVote(self):
        #maps classes to the weight given to that class
        weightedSum = util.Counter()

        #iterates over rows, adds to the class apppearing in row i
        for i in range(self.length):
            row = self.classificationColumn[i]
            weight = self.weights[i]
            weightedSum[row] += weight

        return weightedSum.argMax()


    def computeEntropy(self, weights=None):
        #the column over which entropy is being calculated (the class column)
        column = self.data[:,self.ccIndex]
        #weights default to the general weights (if this is being called within attribute entropy, 
        #weights depend on the value at that attribute)
        if weights is None:
            weights = self.weights

        #number of (weighted) elements in the column
        weightedSize=sum(weights)

        possibleClasses = self.attValues[self.ccIndex]

        #Entropy Value
        H=0.0

        #for each possible class
        for value in possibleClasses:
            #(weighted) number of appearances of this class in colun
            weightedCount=0

            #for each row in the column, if row is equal to the value, add the weight 
            #(this is the dot product of weights with the indicator function on value)
            for i in range(len(column)):   
                if column[i]==value:
                    weightedCount+=weights[i]
            
            #for each value which actually appears in the classification column, 
            #compute the frequency with which it appears and use this for the entropy calculation 
            if weightedCount>0:
                ratio=1.0*weightedCount/weightedSize
                H-=ratio*np.log2(ratio)

        return H

    #boolean indicating if the tree has children
    def isTerminal(self):
        return self.bestAttribute is None


    #Allows for splitting a tree via the given attribute, test avoids some of the computation and is only set to true 
    #when split is used to compute the entropy if the node is selected to be split on
    def split(self, attributeIndex,test=False):  
        #the attribute column                   
        attCol = self.data[:,attributeIndex]
        #possible values attribute actually takes on, needed this instead of all possible values to handle missing
        values=set(attCol)
        values.discard("?")
        # a list to store resulting childre
        newNodes=[]
        #weight to be given to missing values at each leaf
        missingWeight=1.0/len(values)

        #a dictionary giving the new weight to associate to each row after the split
        newWeights={value : [0]*self.length for value in values }

        #for each row, if the value is known, set the value of the corresponding weight vector to 1, and leave others unchanged,
        #if it is missing, split the row equally among the new nodes
        for i in range(self.length):

            if not attCol[i] =="?":
                newWeights[attCol[i]][i] += self.weights[i]

            else:
                for value, newWeight in newWeights.items():
                    newWeights[value][i]+= self.weights[i]*missingWeight
        # a list of the nodes resulting from the split
        newNodes=[ decisionTree(self.data, self.attrsAndValues, self.ccIndex, newWeights[value], value,test) for value in values]
            

        return newNodes


    def bestSplit(self):

        bestAttribute = None
        maxGain = 0

        #get the indices of predictor columns
        colIndices=range(self.data.shape[1])
        colIndices.remove(self.ccIndex)

        #for each column that is not the class column, compute the information gain for splitting along that attribute
        #returns the max/argmax
        for columnIndex in colIndices:
            curEntropy = self.attributeEntropy(columnIndex)
            curGain = self.entropy-curEntropy
            if curGain > maxGain:
                maxGain = curGain
                bestAttribute = self.attributes[columnIndex]
        
        return (maxGain, bestAttribute)


    def attributeEntropy(self, colIndex):
        #Entropy after splitting at attribute H
        attributeH=0.0
        weightedSize=self.weightedSize(colIndex)
        
        #no no non-missing data to split on
        if weightedSize == 0:
            return float("inf")

        #consider splitting the node, compute the expected entropy of the nodes (where the weight is by the size of node)
        potentialSplit=self.split(colIndex,True)


        for node in potentialSplit:
            for i in range(self.length):
                if self.data[i,colIndex]=="?":
                    node.weights[i]=0

            for node in potentialSplit:
                attributeH += 1.0*node.weightedSize(colIndex)/weightedSize*node.entropy
            return attributeH
        else: 
            return float("inf")
    
    #returns the total weight of rows not missing the attribute at colIndex
    def weightedSize(self, colIndex):
        weightedSize=0.0
        
        for i in range(self.length):
            if not self.data[i,colIndex]=="?":
                 weightedSize+=self.weights[i]

        return weightedSize


    #builds a decision tree from data
    def buildTree(self,data):

        tree = Tree(self)

        # stores a leaves that could be split->gain from splitting the leaf
        curLeaves = util.Counter()
        curLeaves[self] = self.bestGain

        print("curLeaves instantiated")

        # if information can still be gained from splitting
        while max(curLeaves.values()) > 0:  # FIXXXXXXXXX
            # choose the leaf that gives the most info. gain, split it and remove 
            # it from possible leaves to be split
            print("largest curLeaves value: " + str(max(curLeaves.values())))
            leafToSplit = curLeaves.argMax()
            print("in while loop, curLeaves.argMax: " + str(leafToSplit.bestAttribute))
            newLeaves = leafToSplit.split(self.attributes.index(leafToSplit.bestAttribute))
            curLeaves.pop(leafToSplit)
            print("popped toSplit")

            #add each new leaf to the list of potential leaves, as well as the corresponding gain if we were to split it
            for leaf in newLeaves:
                print("inside for loop")
                curLeaves[leaf] = leaf.bestGain
                tree.addLeaf(leaf, leafToSplit)
                print("leaf added")

            print("\n")


        return tree

    
    def classify(self, row):

        votes = util.Counter()
        nodes = [(self,1)]

        while len(nodes) > 0:
            
            weightedNode = nodes.pop()
            node = weightedNode[0]
            weight = weightedNode[1]
            
            if node.isTerminal():
                votes[node.majorityVote()] += weight
                
            else:
                bestAttr = node.bestAttribute
                attIndex = self.attributes.index(bestAttr)
                children = node.split(attIndex)

                if row[attIndex] is "?":
                    newWeight = weight/len(children)
                    newNodes = [(child,newWeight) for child in children]
                    nodes.extend(newNodes)
                    break
                
                for child in children:
                    if child.lastSplit==row[attIndex]:
                        nodes.append([child,weight]) 
                        break
           
                
        return node.majorityVote()



def leaveOneOut(data, attr):

    numCorrect = 0

    for i in range(data.shape[1]):
        testRow = data[i,:]
        trainData = np.delete(data, i,0)
        
        DTN = decisionTree(trainData, attr, len(attr) - 1)
        
        classification = DTN.classify(testRow)

        if classification == testRow[len(attr) - 1]:
            numCorrect += 1

    return numCorrect * 1.0 / data.shape[1]



fp = FileParser()
tp = fp.parse_file(sys.argv[1])
data = tp[0]
attrAndVals = tp[1]
DT = decisionTree(data, attrAndVals, len(tp[1]) - 1)
print("built decision tree")
print(DT.buildTree(DT.data).to_string())
row_num = 0
print("Leave one out correctness: " + str(leaveOneOut(data, attrAndVals) * 100) + "% correct.")
