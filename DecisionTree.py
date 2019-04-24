import pandas as pd
import math
import node.py as Node

discrete_features = ['Sex', 'Pclass', 'Embarked']


# Drops useless columns and fills in NaNs with mode for that column
def pre_processing(df):
    df = df.drop(['Name', 'PassengerId', 'Cabin'], axis=1)

    for column in list(df.columns):
        if column != 'Survived':
            df[column].fillna(df.mode()[column][0], inplace=True)

    return df


# Calculates IG for discrete feature
def calculate_ig(df, attribute):
    entropy = calculate_entropy(df)

    attribute_domain = list(df[attribute].unique())

    conditional_entropy = 0

    for x in attribute_domain:
        partition = df.loc[df[attribute] == x]
        conditional_entropy += partition/len(df) * calculate_entropy(partition)

    return entropy - conditional_entropy


# Finds the best threshold and the max IG for continuous attribute
def calculate_ig_cont(df, attribute):
    thresholds = find_thresholds(df, attribute)

    threshold = None
    max_ig = float("-inf")

    for x in thresholds:
        ig = __calculate_ig_cont(df, attribute, x)

        if ig > max_ig:
            threshold = x
            max_ig = ig

    return threshold, ig


# Calculates IG for continuous feature based on a given threshold
def __calculate_ig_cont(df, attribute, threshold):
    entropy = calculate_entropy(df)

    partition_less = df.loc[df[attribute] < threshold]
    partition_greater = df.loc[df[attribute] >= threshold]

    conditional_entropy = len(partition_less)/len(df) * calculate_entropy(partition_less) + \
                          len(partition_greater)/len(df) * calculate_entropy(partition_greater)

    return entropy - conditional_entropy


# Produces a list of thresholds for a continuous attribute
def find_thresholds(df, attribute):
    sorted_df = df[['Survived', attribute]].sort_values('Fare')

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

    return threshold_list


# Calculates the entropy
def calculate_entropy(df):
    count_1 = len(df.loc[df['Survived'] == 1]) / len(df)
    count_0 = len(df.loc[df['Survived'] == 0]) / len(df)

    entropy = -(count_1 * math.log(count_1, 2) + count_0 * math.log(count_0, 2))

    return entropy


# Returns the best attribute to split data on 
def find_best_split(df):
    best_ig = -1
    best_attribute = None
    threshold = False

    for column in list(df.columns):
        if column == 'Survived':
            continue

        if column in discrete_features:
            ig = calculate_ig(column)

            if ig > best_ig:
                best_ig = ig
                best_attribute = column
                threshold = False

        else:
            t, ig = calculate_ig_cont(column)

            if ig > best_ig:
                best_ig = ig
                best_attribute = column
                threshold = t

    return best_attribute, threshold


# Splits the data_frame on highest IG value and returns in format: splits{}, attribute, threshold
def split(df):
    feature, threshold = find_best_split(df)

    splits = {}

    if threshold != False:
        splits['<'] = df.loc[df[feature] < threshold]
        splits['>='] = df.loc[df[feature] >= threshold]

    else:
        for x in list(df[feature].unique()):
            splits[x] = df.loc[df[feature] == x]

    return splits, feature, threshold

# Checks for exit case 2
def check_rows_equal(df):
    for x in list(df.columns):
        if x == 'Survived':
            continue

        if df[x].nunique > 1:
            return False

    return True


def create_tree(df, depth, depth_limit, parent=None):
    if depth >= depth_limit:
        majority_output = df['Survived'].mode()[0]
        return Node(parent, predict_value=majority_output, is_leaf=True)
    if calculate_entropy(df) == 0:
        return Node(parent, predict_value=df['Survived'].iloc[0], is_leaf=True)
    if check_rows_equal(df):
        majority_output = df['Survived'].mode()[0]
        return Node(majority_output, parent, is_leaf=True)

    # Splits dictionary is currently key: df
    splits, feature, threshold = split(df)
    curr_node = Node(parent, feature)

    # Convert splits dictionary to key: node
    for x in splits.keys():
        splits[x] = create_tree(splits[x], depth+1, depth_limit, parent=curr_node)

    curr_node.add_children(splits)

    return curr_node


def train(path, depth_limit):
    df = pd.read_csv(path)
    df = pre_processing(df)

    root = create_tree(df, 0, depth_limit)








