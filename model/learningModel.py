from sklearn import tree
import pandas as pd
import numpy as np
from dtreeviz.trees import dtreeviz
from model import data


class LearningModel(object):
    """整形されたデータを受け取って解析を行う目的です"""

    def __init__(self, train: data.DataModel, test: data.DataModel):
        self.train_data: pd.DataFrame = train.data_frame
        self.test_data: pd.DataFrame = test.data_frame

    # 学習して予想を導く
    def decision_tree_predict(self, target_column_name, feature_column_names, max_depth, min_samples_split):
        target = self.train_data[target_column_name].values
        features_one = self.train_data[feature_column_names].values

        # 決定木の作成
        my_tree_one = tree.DecisionTreeClassifier(
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            random_state=1
        )

        # 学習データをセット
        my_tree_one = my_tree_one.fit(features_one, target)
        test_features = self.test_data[feature_column_names].values
        self.result = my_tree_one.predict(test_features)

        viz = dtreeviz(
            my_tree_one,
            features_one,
            target,
            target_name='variety',
            feature_names=feature_column_names,
            class_names=['Dead', 'Survived'],
        )

        viz.view()

    # 予想結果をCSVに変換して保存
    def convert_result_to_csv(self, path):
        # PassengerIdを取得
        PassengerId = np.array(self.test_data["PassengerId"]).astype(int)
        # my_prediction(予測データ）とPassengerIdをデータフレームへ落とし込む
        my_solution = pd.DataFrame(
            self.result, PassengerId, columns=["Survived"])
        # my_tree_one.csvとして書き出し
        my_solution.to_csv(path, index_label=["PassengerId"])
