from os import add_dll_directory
import numpy as np
import pandas as pd


class DataModel(object):
    """CSV から学習しやすい形に整形する目的です"""

    def __init__(self, csv_path):
        self.data_frame = pd.read_csv(csv_path)
        self.convert_category_to_int()
        self.add_null_datas()

    # 欠損データの確認
    def lack_table(self):
        null_val_column = self.data_frame.isnull().sum()
        percent_column = 100 * self.data_frame.isnull().sum()/len(self.data_frame)
        lack_table = pd.concat([null_val_column, percent_column], axis=1)
        lack_table_ren_columns = lack_table.rename(
            columns={0: 'Lack', 1: '%'})
        return lack_table_ren_columns

    # カテゴリデータを数値に変換する
    def convert_category_to_int(self):
        self.data_frame["Sex"][self.data_frame["Sex"] == "male"] = 0
        self.data_frame["Sex"][self.data_frame["Sex"] == "female"] = 1
        self.data_frame["Embarked"][self.data_frame["Embarked"] == "S"] = 0
        self.data_frame["Embarked"][self.data_frame["Embarked"] == "C"] = 1
        self.data_frame["Embarked"][self.data_frame["Embarked"] == "Q"] = 2

    # nullデータを埋める
    def add_null_datas(self):
        # Ageは中央値
        self.data_frame['Age'] = self.data_frame['Age'].fillna(
            self.data_frame['Age'].median())
        # Embarkedは最頻値
        self.data_frame['Embarked'] = self.data_frame['Embarked'].fillna(
            self.data_frame['Embarked'].max())
        # Frameは中央値
        self.data_frame['Fare'] = self.data_frame.fillna(
            self.data_frame['Fare'].median())
