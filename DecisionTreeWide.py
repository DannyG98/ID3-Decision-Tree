import pandas as pd
import math
import argparse
from node import *

discrete_features = ['Sex', 'Pclass', 'Embarked']
discrete_dict = {}
df_mode = 0


# Drops useless columns and fills in NaNs with mode for that column
def pre_processing(df):
    # df = df.drop(['Name', 'PassengerId', 'Cabin', 'Ticket', 'Age', 'SibSp', 'Parch', 'Fare'], axis=1)
    df = df.drop(['Name', 'PassengerId', 'Cabin', 'Ticket'], axis=1)
    df_mode = df['Survived'].mode()[0]

    for column in list(df.columns):
        if column != 'Survived':
            df[column].fillna(df.mode()[column][0], inplace=True)

        if column in discrete_features:
            discrete_dict[column] = list(df[column].unique())

    return df


# Calculates IG for discrete feature
def calculate_ig(df, attribute, entropy=None):
    if entropy is None:
        entropy = calculate_entropy(df)

    # DICT
    attribute_domain = discrete_dict[attribute]

    conditional_entropy = 0

    for x in attribute_domain:
        partition = df.loc[df[attribute] == x]
        conditional_entropy += len(partition)/len(df) * calculate_entropy(partition)

    return entropy - conditional_entropy


# Finds the best threshold and the max IG for continuous attribute
def calculate_ig_cont(df, attribute, entropy=None):
    thresholds = find_thresholds(df, attribute)

    threshold = None
    max_ig = float("-inf")

    for x in thresholds:
        ig = __calculate_ig_cont(df, attribute, x, entropy)

        if ig > max_ig:
            threshold = x
            max_ig = ig

    return threshold, ig


def calculate_ig_cont_wide(df, attribute, entropy=None):
    if entropy is None:
        entropy = calculate_entropy(df)

    thresholds = list(find_thresholds(df, attribute))

    partitions = [df.loc[df[attribute] < thresholds[0]]]
    for x in range(1, len(thresholds) - 1):
        partition_holder = df.loc[(df[attribute] < thresholds[x]) & (df[attribute] >= thresholds[x-1])]
        partitions.append(partition_holder)

    partitions.append(df.loc[df[attribute] >= thresholds[-1]])

    conditional_entropy = 0

    for x in partitions:
        conditional_entropy += len(x)/len(df) * calculate_entropy(x)

    ig = entropy - conditional_entropy

    return thresholds, ig


# Calculates IG for continuous feature based on a given threshold
def __calculate_ig_cont(df, attribute, threshold, entropy=None):
    if entropy is None:
        entropy = calculate_entropy(df)

    partition_less = df.loc[df[attribute] < threshold]
    partition_greater = df.loc[df[attribute] >= threshold]

    conditional_entropy = len(partition_less)/len(df) * calculate_entropy(partition_less) + \
                          len(partition_greater)/len(df) * calculate_entropy(partition_greater)

    return entropy - conditional_entropy


# Produces a list of thresholds for a continuous attribute
def find_thresholds(df, attribute):
    sorted_df = df[['Survived', attribute]].sort_values(attribute)

    threshold_list = []
    previous_value = sorted_df.iloc[0]['Survived']
    prev_att = sorted_df.iloc[0][attribute]

    for index, row in sorted_df.iterrows():
        if row['Survived'] != previous_value:
            curr_att = row[attribute]

            threshold_list.append(prev_att + (curr_att - prev_att)/2)
            previous_value = row['Survived']

            prev_att = curr_att
        else:
            prev_att = row[attribute]

    return set(threshold_list)


# Calculates the entropy
def calculate_entropy(df):
    if len(df) != 0:
        count_1 = len(df.loc[df['Survived'] == 1]) / len(df)
        count_0 = len(df.loc[df['Survived'] == 0]) / len(df)
    else:
        count_1 = 0
        count_0 = 0

    if count_1 != 0:
        x = count_1 * math.log(count_1, 2)
    else:
        x = 0

    if count_0 != 0:
        y = count_0 * math.log(count_0, 2)
    else:
        y = 0

    entropy = -(x + y)

    return entropy


# Returns the best attribute to split data on
def find_best_split(df):
    best_ig = -1
    best_attribute = None
    threshold = None
    entropy = calculate_entropy(df)

    for column in list(df.columns):
        if column == 'Survived':
            continue

        if column in discrete_features:
            ig = calculate_ig(df, column, entropy)

            if ig > best_ig:
                best_ig = ig
                best_attribute = column
                threshold = None

        else:
            t, ig = calculate_ig_cont_wide(df, column, entropy)

            if ig > best_ig:
                best_ig = ig
                best_attribute = column
                threshold = t

    return best_attribute, threshold


# Splits the data_frame on highest IG value and returns in format: splits{}, attribute, threshold
def split(df):
    feature, threshold = find_best_split(df)

    splits = {}

    if threshold is not None:

        splits[threshold[0]] = df.loc[df[feature] < threshold[0]]#.drop(feature, axis=1)
        for x in range(1, len(threshold) - 1):
            partition_holder = df.loc[(df[feature] < threshold[x]) & (df[feature] >= threshold[x-1])]#.drop(feature, axis=1)
            splits[threshold[x]] = partition_holder

        splits[-1] = df.loc[df[feature] >= threshold[-1]]#.drop(feature, axis=1)

    else:
        # DICT
        for x in discrete_dict[feature]:
            splits[x] = df.loc[df[feature] == x]#.drop(feature, axis=1)

    return splits, feature, threshold


# Checks for exit case 2
def check_rows_equal(df):
    for x in list(df.columns):
        if x == 'Survived':
            continue

        if df[x].nunique() > 1:
            return False

    return True


# Recursively Creates the Decision Tree
def create_tree(df, depth, depth_limit, parent=None):
    try:
        majority_output = df['Survived'].mode()[0]
    except:
        majority_output = df_mode

    if depth >= depth_limit:
        return Node(parent, predict=majority_output, is_leaf=True)
    if calculate_entropy(df) == 0:
        return Node(parent, predict=majority_output, is_leaf=True)
    if check_rows_equal(df):
        return Node(parent, predict=majority_output, is_leaf=True)

    # Splits dictionary is currently key: df
    splits, feature, threshold = split(df)

    continuous = False

    if threshold is not None:
        continuous = True

    curr_node = Node(parent, feature=feature, threshold=threshold, is_continuous=continuous, predict=majority_output)

    # Convert splits dictionary to key: node
    for x in splits.keys():
        splits[x] = create_tree(splits[x], depth+1, depth_limit, parent=curr_node)

    curr_node.add_children(splits)

    return curr_node


# Train a decision tree
def train(data_frame, depth_limit):
    root = create_tree(data_frame, 0, depth_limit)

    return root


# Test a decision tree
def test(model, df):
    current_node = model
    errors = 0

    for index, row in df.iterrows():
        current_node = model

        while not current_node.is_leaf:
            deciding_feature = current_node.feature
            row_feature_val = row[deciding_feature]
            current_node = current_node.get_result_wide(row_feature_val)

        prediction = current_node.predict_value

        if prediction != row['Survived']:
            errors += 1

    return 1 - errors/len(df)


# Self explanatory
def main():
    parser = argparse.ArgumentParser(description='ID3 Decision Tree')
    parser.add_argument("--dataset")
    args = parser.parse_args()
    file_path = args.dataset

    if file_path is None:
        file_path = 'titanic.csv'

    df = pd.read_csv(file_path)
    df = pre_processing(df)
    #df = df.sample(frac=1).reset_index(drop=True)

    split_index = int(len(df) * 0.6)
    train_set = df.iloc[:split_index, :]
    test_set = df.iloc[split_index:, :]

    model = train(train_set, 9)
    print("Training Accuracy:", test(model, train_set))
    print("Test Accuracy: ", test(model, test_set))


if __name__ == '__main__':
    main()
