from DecisionTree import *
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description='ID3 Decision Tree')
parser.add_argument("--dataset", default='titanic.csv')
args = parser.parse_args()
file_path = args.dataset

test_accuracy = {}
train_accuracy = {}

# Read CSV and convert to Pandas dataframe, then do some pre-processing on the data
df = pd.read_csv(file_path)
df = pre_processing(df)

# Randomizes data before splitting
# df = df.sample(frac=1).reset_index(drop=True)

# Data is split 60-40 train, test
split_index = int(len(df) * 0.6)
train_set = df.iloc[:split_index, :]
test_set = df.iloc[split_index:, :]

# Create decision tree of depth 1-15 and log accuracy
for x in range(11):
    model = train(train_set, x)
    emp_err = test(model, train_set)
    test_err = test(model, test_set)

    train_accuracy[x] = emp_err
    test_accuracy[x] = test_err

# Create the lists that contain plots to graph
x_axis = list(test_accuracy.keys())
test_y = [test_accuracy[x] for x in test_accuracy.keys()]
train_y = [train_accuracy[x] for x in train_accuracy.keys()]

# Create and display the graph
plt.plot(x_axis, test_y, label='Test Accuracy', c='Blue')
plt.plot(x_axis, train_y, label='Train Accuracy', c='Red')
plt.legend()
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.grid()
plt.show()