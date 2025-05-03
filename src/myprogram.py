#!/usr/bin/env python
import os
import string
import random
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from data_importer import DataImporter
import dataloader
from model import FFNN
from train import train

UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane
DEVICE = 'cpu'

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        self.model = FFNN(
            input_dim=768,
            hidden_dim=3*768,
            num_classes=UNICODE_BMP_MAX_CODE_POINT
        )

    @classmethod
    def load_training_data(self):
        # your code here
        common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files="common_corpus_10/subset_100_*.parquet")
        common_corpus_stratified = DataImporter.sample_across_languages(common_corpus, minimum_samples=50, sample_size=50)
        print(f'stratified by language corpus size: {common_corpus_stratified.shape}')
        train_dataset, dev_dataset = DataImporter.divide_corpus_into_stratified_datasets(common_corpus_stratified)

        self.train_dataset = train_dataset['text'].tolist()
        self.dev_dataset = dev_dataset['text'].tolist()

    @classmethod
    def load_test_data(cls, fname):
        # your code here
        data = []
        with open(fname) as f:
            for line in f:
                inp = line[:-1]  # the last character is a newline
                data.append(inp)
        return data

    @classmethod
    def write_pred(cls, preds, fname):
        with open(fname, 'wt') as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, work_dir):
        # your code here
        # Create embeddings based on text
        # Create DataLoader to feed in training data
        print("Creating train_dataloader")
        train_cache_path = work_dir + "/train_embeddings"
        train_dataloader = dataloader.create(self.train_dataset, train_cache_path)
        print("Creating dev dataloader...")
        # dev_cache_path = work_dir + "/dev_embeddings.pt"
        # dev_dataloader = dataloader.create(self.dev_dataset, dev_cache_path)
        print("Running training...")
        train_losses = train(
            model=self.model,
            train_dataloader=train_dataloader,
            lr=1e-3,
            n_epochs=10,
            device=DEVICE,
            verbose=True
        )

        print("Final train loss: %.4f", train_losses[-1])
        

    def run_pred(self, data):
        # your code here
        preds = []
        all_chars = string.ascii_letters
        for inp in data:
            # this model just predicts a random character each time
            top_guesses = [random.choice(all_chars) for _ in range(3)]
            preds.append(''.join(top_guesses))
        return preds

    def save(self, work_dir):
        # your code here
        with open(os.path.join(work_dir, 'model.checkpoint'), 'wt') as f:
            torch.save(self.model.state_dict, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        with open(os.path.join(work_dir, 'model.checkpoint')) as f:
            saved_model_state_dict = torch.load(f)
        my_model = MyModel()
        my_model.model.load_state_dict(saved_model_state_dict)


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        MyModel.load_training_data()
        print('Training')
        model.run_train(args.work_dir)
        print('Saving model')
        model.save(args.work_dir)
    elif args.mode == 'test':
        print('Loading model')
        model = MyModel.load(args.work_dir)
        print('Loading test data from {}'.format(args.test_data))
        test_data = MyModel.load_test_data(args.test_data)
        print('Making predictions')
        pred = model.run_pred(test_data)
        print('Writing predictions to {}'.format(args.test_output))
        assert len(pred) == len(test_data), 'Expected {} predictions but got {}'.format(len(test_data), len(pred))
        model.write_pred(pred, args.test_output)
    else:
        raise NotImplementedError('Unknown mode {}'.format(args.mode))
