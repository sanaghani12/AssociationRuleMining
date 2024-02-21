import time
import math
from pyspark.sql import SparkSession
from collections import defaultdict
import os
import sys
os.environ['PYSPARK_PYTHON'] = sys.executable
os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable

class AlgoNEclatClosed:
    def __init__(self):
        self.startTimestamp = 0
        self.endTimestamp = 0
        self.outputCount = 0
        self.numOfFItem = 0
        self.minSupport = 0
        self.item = []
        self.itemsetX = []
        self.itemsetXLen = 0
        self.nlRoot = None
        self.mapItemTIDS = {}
        self.numOfTrans = 0

    def runAlgorithm(self, input_dataset, minsup, output):
        print("runalgo")
        self.nlRoot = SetEnumerationTreeNode()
        self.startTimestamp = time.time()
        spark = SparkSession.builder.master("local[*]").appName("AlgoNEclatClosed").getOrCreate()

        # spark = SparkSession.builder.appName("AlgoNEclatClosed").getOrCreate()

        self.getData(spark, input_dataset, minsup)

        self.itemsetXLen = 0
        self.itemsetX = [0] * self.numOfFItem

        self.buildTree(input_dataset)

        self.nlRoot.label = self.numOfFItem
        self.nlRoot.firstChild = None
        self.nlRoot.next = None

        self.initializeTree()
        self.cpStorage = CPStorage()
        self.outputFilePath = output
        curNode = self.nlRoot.firstChild
        self.nlRoot.firstChild = None
        nextNode = None
        
        while curNode is not None:
            # print("curNode: ",curNode)
            self.traverse(curNode, 1)
            nextNode = curNode.next
            curNode.next = None
            curNode = nextNode

        spark.stop()
        self.endTimestamp = time.time()

    def buildTree(self, filename):
            print("Build tree")
            with open(filename, 'r') as file:
                tid = 1
                for line in file:
                    if line.strip() == '' or line[0] == '#' or line[0] == '%' or line[0] == '@':
                        continue
                    lineSplited = line.split()
                    for itemString in lineSplited:
                        itemX = int(itemString)
                        for j in range(self.numOfFItem):
                            if itemX == self.item[j].index:
                                tids = self.mapItemTIDS.get(j)
                                if tids is None:
                                    tids = set()
                                    self.mapItemTIDS[j] = tids
                                tids.add(tid)
                                break
                    tid += 1
            # Print the populated mapItemTIDS dictionary
            # for item_index, tids in self.mapItemTIDS.items():
            #     print(f"Item index: {item_index}, TIDs: {sorted(tids)}")
   

    def getData(self, spark, filename, minSupport):
        self.numOfTrans = spark.sparkContext.textFile(filename) \
                       .filter(lambda line: line.strip() != '' and line[0] not in '#%@') \
                       .count()
        # Create an RDD from the file, process lines, and flatten the result
        itemsRDD = spark.sparkContext.textFile(filename) \
                    .flatMap(lambda line: [(int(item), 1) for item in line.split()[0:] 
                                            if line.strip() != '' and line[0] not in '#%@'])

        # Reduce by key to count occurrences of each item
        itemCounts = itemsRDD.reduceByKey(lambda a, b: a + b)
        
        # Apply minSupport filter and collect the result
        self.minSupport = math.ceil(minSupport * self.numOfTrans)

        filteredItems = itemCounts.filter(lambda item_count: item_count[1] >= self.minSupport).collect()

        # Process filtered items
        tempItems = [Item(item, count) for item, count in filteredItems]

        self.item = sorted(tempItems, key=lambda x: x.num, reverse=True)
        self.numOfFItem = len(self.item)


    def initializeTree(self):
        lastChild = None
        for t in range(self.numOfFItem - 1, -1, -1):
            nlNode = SetEnumerationTreeNode()
            nlNode.label = t
            nlNode.firstChild = None
            nlNode.next = None
            nlNode.tidSET = self.mapItemTIDS.get(nlNode.label)
            nlNode.count = len(nlNode.tidSET) if nlNode.tidSET is not None else 0

            if self.nlRoot.firstChild is None:
                self.nlRoot.firstChild = nlNode
            else:
                lastChild.next = nlNode
            lastChild = nlNode
    def traverse(self, curNode, level):
        prev = curNode
        siblings = prev.next
        lastChild = None
        sameCount = 0
        
        self.itemsetX[self.itemsetXLen] = curNode.label
        self.itemsetXLen += 1


        while siblings is not None:
            child = SetEnumerationTreeNode()

            if level == 1:
                if curNode.tidSET is not None and len(curNode.tidSET) != 0:
                    child.tidSET = curNode.tidSET - siblings.tidSET
            else:
                if siblings.tidSET is not None and len(siblings.tidSET) != 0:
                    child.tidSET = siblings.tidSET - curNode.tidSET

            child.count = curNode.count - len(child.tidSET) if child.tidSET is not None else 0
            if child.count >= self.minSupport:
                if curNode.count == child.count:
                    self.itemsetX[self.itemsetXLen] = siblings.label
                    self.itemsetXLen += 1
                    sameCount += 1
                else:
                    child.label = siblings.label
                    child.firstChild = None
                    child.next = None
                    if curNode.firstChild is None:
                        curNode.firstChild = child
                    else:
                        lastChild.next = child
                    lastChild = child
            siblings = siblings.next

        itemsetBitset = MyBitVector(self.itemsetX, self.itemsetXLen)
        if self.cpStorage.insertIfClose(itemsetBitset, curNode.count):
            self.writeItemsetsToFile(curNode.count)

        child = curNode.firstChild
        while child is not None:
            self.traverse(child, level + 1)
            child = child.next
        self.itemsetXLen -= (1 + sameCount)

   
    def writeItemsetsToFile(self, support):
        with open(self.outputFilePath, 'a') as file:
            itemset = [str(self.item[i].index) for i in self.itemsetX[:self.itemsetXLen]]
            itemsetStr = ' '.join(itemset)
            file.write(itemsetStr + " #SUP: " + str(support) + "\n")

    def printStats(self):
        print("========== NEclatClosed - STATS ============")
        print("minSupport : " + str(int(100.0 * self.minSupport / self.numOfTrans)) + "%")
        print(" Total time ~: " + str(self.endTimestamp - self.startTimestamp) + " ms")
        print("=====================================")


class Item:
    def __init__(self, index, num):
        self.index = index
        self.num = num


class SetEnumerationTreeNode:
    def __init__(self):
        self.label = 0
        self.firstChild = None
        self.next = None
        self.tidSET = None
        self.count = 0


class MyBitVector:
    TWO_POWER = [2 ** i for i in range(64)]

    def __init__(self, itemset, last):
        if itemset:
            length = itemset[0]
        else:
            print("Error: itemset is empty.")
        self.bits = [0] * ((length // 64) + 1)
        self.cardinality = last
        for i in range(last):
            item = itemset[i]
            self.bits[item // 64] |= MyBitVector.TWO_POWER[item % 64]

    def isSubSet(self, q):
        if self.cardinality >= q.cardinality:
            return False
        for i in range(len(self.bits)):
            if (self.bits[i] & (~q.bits[i])) != 0:
                return False
        return True


class CPStorage:
    def __init__(self):
        self.mapSupportMyBitVector = {}

    def insertIfClose(self, itemsetBitVector, support):
        result = True
        bitvectorList = self.mapSupportMyBitVector.get(support)
        if bitvectorList is None:
            bitvectorList = []
            self.mapSupportMyBitVector[support] = bitvectorList
            bitvectorList.append(itemsetBitVector)
        else:
            index = 0
            for q in bitvectorList:
                if itemsetBitVector.cardinality >= q.cardinality:
                    break
                if itemsetBitVector.isSubSet(q):
                    result = False
                    break
                index += 1
            if result:
                bitvectorList.insert(index, itemsetBitVector)
        return result


def main():
    input_dataset = "D:\DKE\semester 6\Sadeq project dbse\datasets\chess_bk.dat"
    output = "D:\DKE\semester 6\Sadeq project dbse\datasets\closed_chess3.dat"

    minsup = 0.4

    algorithm = AlgoNEclatClosed()

    algorithm.runAlgorithm(input_dataset, minsup, output)
    algorithm.printStats()


if __name__ == "__main__":
    main()
