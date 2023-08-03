# https://huggingface.co/docs/datasets/create_dataset
from datasets import DatasetBuilder, GeneratorBasedBuilder
import datasets
import csv

DATASET_DIR = 'data'

class FTDataset(GeneratorBasedBuilder):
    def _info(self):
        return datasets.DatasetInfo(
            features=datasets.Features({
                'text': datasets.Value('string'),
                'label': datasets.ClassLabel(names=['negative', 'positive', 'neutral']),
            }),
        )
    def _split_generators(self, dl_manager):
        """Returns SplitGenerators."""

        train_path = f'./{DATASET_DIR}/train.csv'
        test_path = f'./{DATASET_DIR}/test.csv'
        eval_path = f'./{DATASET_DIR}/eval.csv'

        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN, gen_kwargs={"filepath": train_path}),
            datasets.SplitGenerator(name=datasets.Split.TEST, gen_kwargs={"filepath": test_path}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION, gen_kwargs={"filepath": eval_path}),
        ]

    def _generate_examples(self, filepath):
        # CSVファイルを行ごとに読み込み、それぞれの行をHugging Faceデータセットの形式に変換
        with open(filepath, encoding='utf-8') as csv_file:
            csv_reader = csv.reader(csv_file)
            for id_, row in enumerate(csv_reader):
                yield id_, {
                    'text': row[0], 
                    'label': row[1], 
                }