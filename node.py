class Node:
    def __init__(self, parent, feature=None, predict=0, children={}, threshold=None, is_continuous=False, is_leaf=False):

        if is_leaf:
            self.is_leaf = True
        if not is_leaf:
            self.is_leaf = False

        self.predict_value = predict
        self.feature = feature
        self.parent = parent
        self.children = children
        self.is_continous = is_continuous
        self.threshold = threshold

    # Adds the children dictionary
    def add_children(self, children_dict):
        self.children = children_dict

    # This method is used for DecisionTree.py only
    # Returns the result of a decision from a decision stump.
    # Can return either a yes/no value or another decision stump
    def get_result(self, evaluate_on):
        if self.is_leaf:
            return self.predict_value

        if self.is_continous:
            if evaluate_on < self.threshold:
                return self.children['<']
            if evaluate_on >= self.threshold:
                return self.children['>=']
        else:
            return self.children[evaluate_on]

    # This method is used for DecisionTreeWide.py only
    # Returns the result of a decision from a decision stump.
    # Can return either a yes/no value or another decision stump
    def get_result_wide(self, evaluate_on):
        if self.is_leaf:
            return self.predict_value

        if self.is_continous:
            keys = list(self.children.keys())
            keys.sort()

            for x in keys:
                if evaluate_on < x:
                    return self.children[x]

            return self.children[-1]

        else:
            return self.children[evaluate_on]