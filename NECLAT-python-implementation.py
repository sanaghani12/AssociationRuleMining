import sys
import time
import math
import heapq
import random
import itertools
import numpy as np
from collections import defaultdict

class AlgoNEclatClosed:
    def __init__(self):
        self.startTimestamp = 0
        self.endTimestamp = 0
        self.outputCount = 0
        self.writer = None
        self.numOfFItem = 0
        self.minSupport = 0
        self.item = []
        self.itemsetX = []
        self.itemsetXLen = 0
        self.nlRoot = None
        self.mapItemTIDS = {}
        self.cpStorage = None
        self.comp = lambda a, b: b.num - a.num
        self.numOfTrans = 0

    def runAlgorithm(self, input_dataset, minsup, output):
        print("runalgo")
        self.nlRoot = SetEnumerationTreeNode()
        self.writer = open(output, 'w')
        self.startTimestamp = time.time()

        self.getData(input_dataset, minsup)

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

        self.writer.close()
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
        for item_index, tids in self.mapItemTIDS.items():
            print(f"Item index: {item_index}, TIDs: {sorted(tids)}")

    def getData(self, filename, minSupport):
        self.numOfTrans = 0
        mapItemCount = defaultdict(int)
        with open(filename, 'r') as file:
            for line in file:
                if line.strip() == '' or line[0] == '#' or line[0] == '%' or line[0] == '@':
                    continue
                self.numOfTrans += 1
                lineSplited = line.split()
                for itemString in lineSplited:
                    item = int(itemString)
                    mapItemCount[item] += 1
            
        self.minSupport = math.ceil(minSupport * self.numOfTrans)
        print("self.numOfTrans: ",self.numOfTrans)
        tempItems = []
        print("mapItemCount items length: ", len(mapItemCount.items()))
        print("mapItemCount items: ", mapItemCount.items())
        for item, count in mapItemCount.items():
            if count >= self.minSupport:
                tempItems.append(Item(item, count))

        self.item = sorted(tempItems, key=lambda x: x.num, reverse=True)
        self.numOfFItem = len(self.item)
        # Printing each item in self.item
        print("Frequent Items:")
        for item in self.item:
            print(f"Item ID: {item.index}, Count: {item.num}")
        # print("self.item ",tempItems )
        print("self.numOfFItem ",self.numOfFItem )

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
                lastChild = nlNode
            else:
                lastChild.next = nlNode
                lastChild = nlNode

    def traverse(self, curNode, level):
        prev = curNode
        sibling = prev.next
        lastChild = None
        sameCount = 0
        # print("curNode.label in Traverse: ",curNode.label)
        self.itemsetX[self.itemsetXLen] = curNode.label
        self.itemsetXLen += 1
        while sibling is not None:
            child = SetEnumerationTreeNode()

            if level == 1:
                if curNode.tidSET is not None and len(curNode.tidSET) != 0:
                    child.tidSET = curNode.tidSET - sibling.tidSET
            else:
                if sibling.tidSET is not None and len(sibling.tidSET) != 0:
                    child.tidSET = sibling.tidSET - curNode.tidSET

            child.count = curNode.count - len(child.tidSET) if child.tidSET is not None else 0
            if child.count >= self.minSupport:
                if curNode.count == child.count:
                    self.itemsetX[self.itemsetXLen] = sibling.label
                    self.itemsetXLen += 1
                    sameCount += 1
                else:
                    child.label = sibling.label
                    child.firstChild = None
                    child.next = None
                    if curNode.firstChild is None:
                        curNode.firstChild = child
                        lastChild = child
                    else:
                        lastChild.next = child
                        lastChild = child
            sibling = sibling.next

        itemsetBitset = MyBitVector(self.itemsetX, self.itemsetXLen)
        if self.cpStorage.insertIfClose(itemsetBitset, curNode.count):
            # print("curNode.count: ", curNode.count)
            self.writeItemsetsToFile(curNode.count)

        child = curNode.firstChild
        nextNode = None
        curNode.firstChild = None
        while child is not None:
            self.traverse(child, level + 1)
            nextNode = child.next
            child.next = None
            child = nextNode
        self.itemsetXLen -= (1 + sameCount)


    def writeItemsetsToFile(self, support):
        itemset = [str(self.item[i].index) for i in self.itemsetX[:self.itemsetXLen]]
        itemsetStr = ' '.join(itemset)
        self.writer.write(itemsetStr + " #SUP: " + str(support) + "\n")

    def printStats(self):
        print("hai")
        print("========== NEclatClosed - STATS ============")
        print("minSupport : " + str(int(100.0 * self.minSupport / self.numOfTrans)) + "%")
        print(" Total time ~: " + str(self.endTimestamp - self.startTimestamp) + " ms")
        print(" Max memory:" + str(self.getMaxMemory()) + " MB")
        print("=====================================")

    def getMaxMemory(self):
        return round(sys.getsizeof(self) / (1024 * 1024), 2)


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
        length = itemset[0]
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
    # input_dataset = "C:/Users/acer/Documents/DBSE Project/New folder/chess.dat"
    print("hai")
    # output = "C:/Users/acer/Documents/DBSE Project/New folder/closed_chess.dat"
    output = "D:\DKE\semester 6\Sadeq project dbse\datasets\closed_chess.dat"

    minsup = 0.4

    algorithm = AlgoNEclatClosed()

    algorithm.runAlgorithm(input_dataset, minsup, output)
    algorithm.printStats()


if __name__ == "__main__":
    main()