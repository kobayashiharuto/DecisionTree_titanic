from sklearn import tree
import pandas as pd
import numpy as np

from model import data


class LearningModel(object):
    """整形されたデータを受け取って解析を行う目的です"""

    def __init__(self, train_data: data.DataModel, test_data: data.DataModel):
        self.train_data: pd.DataFrame = train_data.data_frame
        self.test_data: pd.DataFrame = test_data.data_frame

    def decision_tree_predict(self, feature_datas):
        target = self.train_data['Survived'].values
        features_one = self.train_data[feature_datas].values
        # 決定木の作成
        my_tree_one = tree.DecisionTreeClassifier()
        my_tree_one = my_tree_one.fit(features_one, target)
        # 「test」の説明変数の値を取得
        test_features = self.test_data[feature_datas].values
        # 「test」の説明変数を使って「my_tree_one」のモデルで予測
        self.result = my_tree_one.predict(test_features)

    def convert_result_to_csv(self, path):
        # PassengerIdを取得
        PassengerId = np.array(self.test_data["PassengerId"]).astype(int)
        # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
        my_solution = pd.DataFrame(
            self.result, PassengerId, columns=["Survived"])
        # my_tree_one.csvとして書き出し
        my_solution.to_csv(path, index_label=["PassengerId"])
        print('success!')
