from DecisionTreeWide import *
import matplotlib.pyplot as plt

test_accuracy = {}
train_accuracy = {}

df = pd.read_csv('titanic.csv')
df = pre_processing(df)

split_index = int(len(df) * 0.6)
train_set = df.iloc[:split_index, :]
test_set = df.iloc[split_index:, :]

for x in range(15):
    model = train(train_set, x)
    emp_err = test(model, train_set)
    test_err = test(model, test_set)

    train_accuracy[x] = emp_err
    test_accuracy[x] = test_err


x_axis = list(test_accuracy.keys())
test_y = [test_accuracy[x] for x in test_accuracy.keys()]
train_y = [train_accuracy[x] for x in train_accuracy.keys()]

plt.plot(x_axis, test_y, label='Test Accuracy', c='Blue')
plt.plot(x_axis, train_y, label='Train Accuracy', c='Red')
plt.legend()
plt.xlabel('Tree Depth')
plt.ylabel('Accuracy')
plt.grid()
plt.show()