#encoding: utf-8

print(__doc__)
import operator
from math import log
import decisionTreePlot as dtPlot
from collections import Counter

def createDataSet():
    '''
    定义数据集
    :return: 标签和数据集
    '''
    dataSet = [[1,1,'yes'],
               [1,1,'yes'],
               [1,0,'no'],
               [0,1,'no'],
               [0,1,'no']]
    labels = ['no surfacing', 'flippers']
    return dataSet, labels

def calcShannonEnt(dataSet):
    '''
    计算香农熵
    :param dataSet: 数据集
    :return: 香农熵
    '''
    numEntries = len(dataSet)# 计算数据集长度
    labelCounts = {}# label出现的次数

    for featVec in dataSet:
        # 储存标签
        currentLabel = featVec[-1]
        # 为所有可能的分类创建字典，如果发现当前键值不存在则加入字典
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1

        # 求label标签的香农熵
        shannonEnt = 0.0
        for key in labelCounts:
            # 计算label出现的概率
            prob = float(labelCounts[key])/numEntries

            # 计算香农熵
            shannonEnt -= prob*log(prob,2)
        return shannonEnt

# 划分数据集
def splitDataSet(dataSet, index, value):
    '''
    通过遍历来求出index对应的column列为value的行

    :param dataSet: 待划分的数据集
    :param index: 划分数据集的特征
    :param value: 需要返回的特征值
    :return: index 列为value的数据集（去除index列）
    '''
    retDataSet = []
    for featVec in dataSet:
        if featVec[index] == value:
            # 取前index行
            reducedFeatVec = featVec[:index]
            reducedFeatVec.extend(featVec[index+1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet

def chooseBestFeatureSplit(dataSet):
    '''

    :param dataSet:
    :return: bestFeature:最优特征列

    '''
    # 求第一行有多少列的特征，去除最后一列标签
    numFeatures = len(dataSet[0])-1
    # 计算原始信息熵
    baseEntropy = calcShannonEnt(dataSet)
    bestInfoGain, bestFeature = 0.0, -1

    for i in range(numFeatures):
        # 获取对应feature下的全部数据
        featlist = [example[i] for example in dataSet]
        # 使用set 对数据集去重
        uniqueVals = set(featlist)

        newEntropy = 0.0
        # 遍历
        for value in uniqueVals:
            subDataSet = splitDataSet(dataSet, i, value)
            # 计算概率
            prob = len(subDataSet)/float(len(dataSet))
            # 计算信息熵
            newEntropy += prob * calcShannonEnt(subDataSet)
        # 信息增益
        infoGain = baseEntropy - newEntropy
        print('infoGain=', infoGain, 'bestFeature=', i, baseEntropy, newEntropy)
        if (infoGain > bestInfoGain):
            bestInfoGain = infoGain
            bestFeature = 1

    return bestFeature

def majorityCnt(classList):
    '''
    选择出现次数最多的一个结果
    :param classList: 类（label）的集合
    :return: 最优特征列bestFeaure

    '''
    classCount = {}
    for vote in classList:
        if vote not in classList.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    # 倒序排列，取第一个即为结果
    sortedClassCount = sorted(classCount.iteritems(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]



def createTree(dataSet, labels):
    '''
    构造树的数据结构
    :param dataSet:
    :param labels:
    :return: myTree
    '''
    classList = [example[-1] for example in dataSet]
    # 停止条件：1 只有一个类别，直接返回结果：最后一列出现的次数等于集合的长度
    if classList.count(classList[0]) == len(classList):
        return classList[0]
    # 2：
    if len(dataSet[0]) == 1:
        return majorityCnt(classList)

    # 选择最优特征列，得到最优特征列的标签
    bestFeature = chooseBestFeatureSplit(dataSet)
    bestFeaturelabel = labels[bestFeature]

    # 初始化mytree
    myTree = {bestFeaturelabel: {}}
    #重要
    del labels[bestFeature]
    # 取出最优列，然后他的branch做分类
    featValues = [example[bestFeature] for example in dataSet]
    uniqueVals = set(featValues)
    for value in uniqueVals:
        # 求出剩余标签label
        sublabels = labels[:]
        '''遍历当前选择特征包含的所有属性值，在每一个数据集划分上使用createtree（）'''
        myTree[bestFeaturelabel][value] = createTree(splitDataSet(dataSet,bestFeature,value),sublabels)

    return myTree

def classify(inputTree, featLabels, testVec):
    '''
    给定输入的节点进行分类
    :param inputTree: 决策树模型
    :param featLabels: 特征标签对应的名字
    :param testVec: 输入的测试集
    :return: classLbels: 分类的结果,需要映射label
    '''
    # 获取tree的根节点对应的key
    firstStr = inputTree.keys()[0]
    # 通过key得到根节点对应的value
    secondDict = inputTree[firstStr]
    # 判断根节点
    featIndex = featLabels.index(firstStr)

    key = testVec[featIndex]
    value0fFeat = secondDict[key]
    print('+++', firstStr, 'xxx', secondDict, '---', key, '>>>', value0fFeat)
    # 判断分支是否结束： 条件是判断value0fFeat是否为字典
    if isinstance(value0fFeat, dict):
        classLabel = classify(value0fFeat, featLabels, testVec)
    else:
        classLabel = value0fFeat
    return classLabel





