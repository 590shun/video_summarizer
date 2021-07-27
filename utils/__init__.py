import os
import json

def parse_splits_filename(splits_filename):
    # Parse split file and count number of k_folds
    _, sfname = os.path.split(splits_filename)
    sfname, _ = os.path.splitext(sfname)
    # データセット名を取得(e.g. TVSum)
    dataset_name = sfname.split('_')[0]

    # 各split.jsonファイル内の分割数を取得
    with open(splits_filename, 'r') as sf:
        splits = json.load(sf)

    # dataset_name: summe, tvsum, etc
    # splits: [{"train_keys": [...], "test_keys": [...]}, ...]
    # with len(splits) := number of k_folds
    return dataset_name, splits


class Proportion(object):
    # 0-1の比率を表す制限されたfloat class
    def __eq__(self, value):
        return 0 < value <= 1

    def __contains__(self, item):
        return self.__eq__(item)

    def __iter__(self):
        yield 'a proportion value in ]0, 1]'

    def __str__(self):
        return 'a proportion value in ]0, 1]'
