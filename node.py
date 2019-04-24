class Node:
    def __init__(self, parent, value, predict=0, children={}, is_continuous=False, is_leaf=False):

        if is_leaf:
            self.is_leaf = True
            self.value = value
            self.predict_value = predict
        if not is_leaf:
            self.is_leaf = False

        self.value = value
        self.parent = parent
        self.children = children
        self.is_continous = is_continuous

    def add_children(self, children_dict):
        self.children = children_dict

    def get_result(self, evaluate_on):
        if self.is_leaf:
            return self.predict_value

        if self.is_continous:
            if evaluate_on < self.value:
                return self.data_dict['<']
            if evaluate_on >= self.value:
                return self.data_dict['>=']
        else:
            return self.data_dict[evaluate_on]
