import pandas as pd
from data_importer import DataImporter
import dataloader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from myprogram import MyModel
from pprint import pprint
from torchmetrics import Precision, Recall, F1Score, Accuracy
import torch
import random
import string

UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane

def generate_test_data(data_files="common_corpus_10/subset_1_1*.parquet", records=10000, random_state=42, only_characters=False):
    print('only characters:', only_characters)
    common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files=data_files)
    common_corpus_sampled = common_corpus.sample(n=records, random_state=random_state)
    samples = common_corpus_sampled['text'].tolist()

    x = []
    y = []
    for sample in samples:
        sample = sample.replace('\n', ' ')
        if only_characters:
            translator = str.maketrans('', '', string.punctuation + string.digits)
            sample = sample.translate(translator)
        valid_indices = [ i for i in range(len(sample[:100])-1) if (i+1 < len(sample)) and sample[i+1] != ' ' ]
        if len(valid_indices) == 0:
            continue

        length = random.choice(valid_indices)
        x.append(sample[:length])
        y.append(sample[length+1])
    return x, y

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to load the model from', default='work')
    parser.add_argument('--test_output', help='path to write entire output', default='work/testing_output.csv')
    parser.add_argument('--only_characters', help='remove punctuation and numbers from the test dataset', default=False, action='store_true')
    args = parser.parse_args()

    print('Loading model')
    model = MyModel.load(args.work_dir)

    print('Generating test data')
    test_data, labels = generate_test_data(only_characters=args.only_characters)
    print('Making predictions')
    pred = model.run_pred(test_data, verbose=True)

    is_correct = [labels[i] in pred[i] for i in range(len(pred))]

    accuracy = sum(is_correct) / len(is_correct)
    print(f'{sum(is_correct)} of {len(is_correct)} correct.')
    print(f'accuracy: {accuracy:.4f}')

    final_results = pd.DataFrame({
        'text': test_data,
        'expected_label': labels,
        'predicted_labels': pred,
        'is_correct': is_correct,
    })

    print('Writing predictions to {}'.format(args.test_output))
    final_results.to_csv(args.test_output, index=False)
