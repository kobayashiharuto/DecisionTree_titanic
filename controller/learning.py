from model import data, learningModel


def learning():
    # データモデルの作成
    train_data = data.DataModel('data/train.csv')
    test_data = data.DataModel('data/test.csv')

    train_data.convert_category()
    test_data.convert_category()

    train_data.add_null_datas()
    test_data.add_null_datas()

    # 学習モデルの作成
    learner = learningModel.LearningModel(
        train_data=train_data, test_data=test_data)
    learner.decision_tree_predict(
        ['Pclass', 'Sex', 'Age', 'Fare', 'SibSp', 'Parch', 'Embarked'], max_depth=10, min_samples_split=5)
    learner.convert_result_to_csv('out/result.csv')
