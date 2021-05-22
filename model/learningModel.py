from sklearn import tree


class LearningModel(object):
    """整形されたデータを受け取って解析を行う目的です"""

    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
