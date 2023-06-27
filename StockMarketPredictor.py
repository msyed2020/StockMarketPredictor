import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib as plt

# Initialize data sets
GOLD_TRAIN_DATA = 'Gold Data Last Year.csv'
GOLD_TEST_DATA = 'Gold Data Last Month.csv'
GAS_TRAIN_DATA = 'Gas Data Last Year.csv'
GAS_TEST_DATA = 'Gas Data Last Month.csv'
OIL_TRAIN_DATA = 'Oil Data Last Year.csv'
OIL_TEST_DATA = 'Oil Data Last Month.csv'
SILVER_TRAIN_DATA = 'Silver Data Last Year.csv'
SILVER_TEST_DATA = 'Silver Data Last Month.csv'

# data sets we actually use
currentTrainData = GOLD_TRAIN_DATA
currentTestData = GOLD_TEST_DATA
# Num of data points to retrieve from csv files
NUM_TRAIN_DATA_POINTS = 266
NUM_TEST_DATA_POINTS = 22

LEARNING_RATE = 0.1
NUM_EPOCHS = 100

def loadStockData(stockName, numDataPoints):
    data = pd.read_csv(stockName, skiprows=0, nrows=numDataPoints, usecols=['Price', 'Open', 'Vol.'])
    # Final price of stock
    finalPrice = data['Price'].astype(str).str.replace(',', '').astype(float)
    # Opening price of stock
    openingPrice = data['Open'].astype(str).str.replace(',', '').astype(float)
    # Volume of stock traded throughout the day
    volume = data['Vol.'].str.strip('MK').astype(float)
    return finalPrice, openingPrice, volume

def calculatePriceDifferences(finalPrice, openingPrice):
    priceDifference = []
    for i in range(len(finalPrice)-1):
        priceDiff = openingPrice[i + 1] - finalPrice[i]
        priceDifference.append(priceDiff)
    return priceDifference

def calculateAccuracy(expected_values, actual_values):
    num_correct = 0
    for a_i in range(len(actual_values)):
        if actual_values[a_i] < 0 < expected_values[a_i]:
            num_correct += 1
        elif actual_values[a_i] > 0 > expected_values[a_i]:
            num_correct += 1
    return (num_correct / len(actual_values)) * 100


trainFinalPrices, trainOpeningPrices, trainVolumes = loadStockData(currentTrainData, NUM_TRAIN_DATA_POINTS)
trainPriceDifferences = calculatePriceDifferences(trainFinalPrices, trainOpeningPrices)
trainVolumes = trainVolumes[:-1]

testFinalPrices, testOpeningPrices, testVolumes = loadStockData(currentTestData, NUM_TRAIN_DATA_POINTS)
testPriceDifferences = calculatePriceDifferences(testFinalPrices, testOpeningPrices)
testVolumes = testVolumes[:-1]


finals = loadStockData(currentTestData, NUM_TEST_DATA_POINTS)
openings = loadStockData(currentTestData, NUM_TEST_DATA_POINTS)
volumes = loadStockData(currentTestData, NUM_TEST_DATA_POINTS)
print(calculatePriceDifferences(finals, openings))

x = tf.placeholder(tf.float32, name='x')
W = tf.Variable([.1], name="W")
b = tf.Variable([.1], name="b")
y = tf.Variable([.1], name="y")
yActual = W * x + b
yPredicted = tf.placeholder(tf.float32, name='yPredicted')
loss = tf.reduce_sum(tf.square(y - yPredicted))
optimizer = tf.train.AdamOptimizer(LEARNING_RATE).minimize(loss)

session = tf.Session()
session.run(tf.global_variables_initializer())

for _ in range(NUM_EPOCHS):
    session.run(optimizer, feed_dict={x: trainVolumes, yPredicted: trainPriceDifferences})

results = session.run(y, feed_dict={x: testVolumes})
accuracy = calculateAccuracy(testPriceDifferences, results)
print("Accuracy of model: {0:.2f}%".format(accuracy))

plt.figure(1)
plt.plot(trainVolumes, trainPriceDifferences)
plt.title('Price Differences for Given Volumes of Past Year')
plt.xlabel('Volumes')
plt.ylabel('Price Differences')
plt.show()

