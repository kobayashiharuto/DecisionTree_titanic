from model import data


def learning():
    train_data = data.DataModel('data/test.csv')
    test_data = data.DataModel('data/train.csv')

    train_data.convert_category()
    test_data.convert_category()

    train_data.add_null_datas()
    test_data.add_null_datas()
