from model import data, learningModel


def learning():
    train_data = data.DataModel('data/train.csv')
    test_data = data.DataModel('data/test.csv')

    train_data.convert_category()
    test_data.convert_category()

    train_data.add_null_datas()
    test_data.add_null_datas()

    # 学習モデルの作成
    learner = learningModel.LearningModel(
        train_data=train_data, test_data=test_data)
    result = learner.decision_tree_predict()
    print(result)
