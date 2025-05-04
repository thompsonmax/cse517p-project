#!/usr/bin/env python
import os
import string
import random
import torch.nn as nn
import torch
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
import pandas as pd
from data_importer import DataImporter
import dataloader
from model import FFNN
from train import train
from predict import predict
from pprint import pprint
import errno

UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

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

    def mkdir(self, work_dir):
        try:
            os.mkdir(work_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @classmethod
    def load_training_data(self, work_dir, force=False):
        train_dir = work_dir + "/train_embeddings"
        dev_dir = work_dir + "/dev_embeddings"
        X_train_path = work_dir + "/train_embeddings/x_embeddings.pt"
        y_train_path = work_dir + "/train_embeddings/y_embeddings.pt"
        X_dev_path = work_dir + "/dev_embeddings/x_embeddings.pt"
        y_dev_path = work_dir + "/dev_embeddings/y_embeddings.pt"

        if os.path.isdir(train_dir) and os.path.isdir(dev_dir) and not force:
            self.X_train = torch.load(X_train_path)
            self.y_train = torch.load(y_train_path)
            self.X_dev = torch.load(X_dev_path)
            self.y_dev = torch.load(y_dev_path)
            return
        common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files="common_corpus_10/subset_100_*.parquet")
        common_corpus_stratified = DataImporter.sample_across_languages(common_corpus, minimum_samples=50, sample_size=50)
        print(f'stratified by language corpus size: {common_corpus_stratified.shape}')
        train_dataset, dev_dataset = DataImporter.divide_corpus_into_stratified_datasets(common_corpus_stratified)

        train_dataset = train_dataset['text'].tolist()
        dev_dataset = dev_dataset['text'].tolist()

        print('preparing training dataset')
        self.X_train, self.y_train = dataloader.create(train_dataset, device=DEVICE)
        print('preparing dev dataset')
        self.X_dev, self.y_dev = dataloader.create(dev_dataset, device=DEVICE)
        self.mkdir(self, train_dir)
        self.mkdir(self, dev_dir)
        torch.save(self.X_train, X_train_path)
        torch.save(self.y_train, y_train_path)
        torch.save(self.X_dev, X_dev_path)
        torch.save(self.y_dev, y_dev_path)

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
        train_losses, final_dev_metrics = train(
            model=self.model,
            X_train=self.X_train,
            y_train=self.y_train,
            X_dev=self.X_dev,
            y_dev=self.y_dev,
            lr=1e-3,
            n_epochs=10,
            device=DEVICE,
            verbose=True,
        )

        print("Final train loss: %.4f" % (train_losses[-1]))
        print("Final dev metrics:")
        pprint(final_dev_metrics)
        

    def run_pred(self, data):
        # your code here
        # test_cache_path = 
        X_test = dataloader.create_test(data)
        preds = predict(
            X_test,
            self.model,
            device=DEVICE
        )
        return preds

    def save(self, work_dir):
        # your code here
        model_path = os.path.join(work_dir, 'model.checkpoint')
        torch.save(model.model.state_dict(), model_path)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model_path = os.path.join(work_dir, 'model.checkpoint')
        saved_model_state_dict = torch.load(model_path)
        my_model = MyModel()
        my_model.model.load_state_dict(saved_model_state_dict)
        return my_model


if __name__ == '__main__':
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('mode', choices=('train', 'test'), help='what to run')
    parser.add_argument('--work_dir', help='where to save', default='work')
    parser.add_argument('--test_data', help='path to test data', default='example/input.txt')
    parser.add_argument('--test_output', help='path to write test predictions', default='pred.txt')
    parser.add_argument('--force', type=bool, help='rebuild the embeddings, even if a saved version is available', default=False)
    args = parser.parse_args()

    random.seed(0)

    if args.mode == 'train':
        if not os.path.isdir(args.work_dir):
            print('Making working directory {}'.format(args.work_dir))
            os.makedirs(args.work_dir)
        print('Instatiating model')
        model = MyModel()
        print('Loading training data')
        MyModel.load_training_data(args.work_dir, force=args.force)
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
