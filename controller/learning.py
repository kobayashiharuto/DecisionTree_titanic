from model import data, learningModel


def learning():
    # データモデルの作成
    train_data = data.DataModel('data/train.csv')
    test_data = data.DataModel('data/test.csv')

    # 学習モデルの作成
    learner = learningModel.LearningModel(
        train=train_data,
        test=test_data
    )
    # パラメータを渡して予想する
    learner.decision_tree_predict(
        target_column_name='Survived',
        feature_column_names=[
            'Pclass',
            'Sex',
            'Age',
            'Fare',
            'SibSp',
            'Parch',
        ],
        max_depth=10,
        min_samples_split=5
    )
    # 結果をCSVに吐き出す
    learner.convert_result_to_csv('out/result.csv')
    # 結果を
    print('success!')
