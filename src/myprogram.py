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
import transformer_model
import train
from predict import predict, predict_transformer
from pprint import pprint
import errno
import hyperparams
import pickle
import streaming_dataloader

# UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device: " + DEVICE)

class MyModel:
    """
    This is a starter model to get you started. Feel free to modify this file.
    """

    def __init__(self):
        print(f"Using pytorch version: {torch.__version__}")
        self.model = transformer_model.CharacterTransformer(
            embed_dim=hyperparams.EMBED_DIM,
            nhead=hyperparams.N_HEADS,
            num_decoder_layers=hyperparams.N_DECODER_LAYERS,
            dim_feedforward=hyperparams.FF_DIM,
            dropout=hyperparams.DROPOUT_RATE,
        ).to(DEVICE)

    def mkdir(self, work_dir):
        try:
            os.mkdir(work_dir)
        except OSError as e:
            if e.errno != errno.EEXIST:
                raise

    @classmethod
    def load_training_data(self, work_dir, force=False):
        vocab_dir = os.path.join(work_dir, 'vocab')
        vocab_path = os.path.join(vocab_dir, 'vocab.pt')

        if os.path.isdir(vocab_dir) and not force:
            with open(vocab_path, 'rb') as f:
                self.char_vocab = pickle.load(f)
            print(f"Vocab 3: {chr(self.char_vocab[3])}")
            print(f"Vocab 4: {chr(self.char_vocab[4])}")
            print(f"Vocab 8: {chr(self.char_vocab[8])}")
            print(f"Vocab 11: {chr(self.char_vocab[11])}")
            print(f"Vocab 14: {chr(self.char_vocab[14])}")
        else:
            common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files="common_corpus_10/subset_100_*.parquet")
            common_corpus_stratified = DataImporter.sample_across_languages(common_corpus, minimum_samples=2, max_samples=hyperparams.DATASET_MAX_SAMPLES)
            print(f'stratified by language corpus size: {common_corpus_stratified.shape}')
            print(f'stratfied data head {common_corpus_stratified[10:]}')
            # train_dataset, dev_dataset = DataImporter.divide_corpus_into_datasets(common_corpus_stratified)

            data = common_corpus_stratified['text'].tolist()
            self.char_vocab = dataloader.preprocess_transformer(data, device=DEVICE)
            # print(f'X_train len: {len(self.X_train)}')
            # print(f'y_train len: {len(self.y_train)}')
            # print('preparing dev dataset')
            # self.X_dev, self.y_dev, _ = dataloader.preprocess_transformer(dev_dataset, device=DEVICE, char_vocab=self.char_vocab)
            # print(f'X_dev len: {len(self.X_dev)}')
            # print(f'y_dev len: {len(self.y_dev)}')
            print("Saving vocab data...")
            self.mkdir(self, vocab_dir)
            with open(vocab_path, 'wb') as f:
                pickle.dump(self.char_vocab, f)
        
        print(f"Vocabulary has size {len(self.char_vocab)}")
        # Create streaming dataloader given vocab
        print("Creating streaming dataloader...")
        vocab2idx = {v: k for k, v in enumerate(self.char_vocab)}
        self.streaming_dataloader = streaming_dataloader.create_streaming_dataloader(vocab2idx)
        
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
        with open(fname, 'wt', encoding="utf-8") as f:
            for p in preds:
                f.write('{}\n'.format(p))

    def run_train(self, work_dir):
        # print('x_train len: {}'.format(len(self.X_train)))
        # your code here
        self.model.init_with_vocab(self.char_vocab)
        train_losses, final_dev_metrics = train.train_transformer(
            model=self.model,
            dataloader=self.streaming_dataloader,
            vocab_size=len(self.char_vocab),
            work_dir=work_dir,
            lr=1e-3,
            n_epochs=10,
            device=DEVICE,
            verbose=True,
        )

        print("Final train loss: %.4f" % (train_losses[-1]))
        print("Final dev metrics:")
        print(final_dev_metrics)
        

    def run_pred(self, data, verbose=False):
        # your code here
        # test_cache_path = 
        if self.char_vocab is None:
            raise Exception("vocab isn't set!")

        #X_test = dataloader.preprocess_transformer(data, device=DEVICE, char_vocab=self.char_vocab)
        X_test = data
        preds = predict_transformer(
            X_test,
            self.model,
            vocab=self.char_vocab,
            device=DEVICE,
            verbose=verbose,
        )
        return preds

    def save(self, work_dir):
        # your code here
        model_path = os.path.join(work_dir, 'model.checkpoint')
        vocab_path = os.path.join(work_dir, 'vocab.pt')
        torch.save(self.model.state_dict(), model_path)
        with open(vocab_path, 'wb') as f:
            pickle.dump(self.char_vocab, f)

    @classmethod
    def load(cls, work_dir):
        # your code here
        # this particular model has nothing to load, but for demonstration purposes we will load a blank file
        model_path = os.path.join(work_dir, 'model.checkpoint')
        print(f'Model path: {model_path}')
        saved_model_state_dict = torch.load(model_path, map_location=DEVICE)
        my_model = MyModel()
        vocab_path = os.path.join(work_dir, 'vocab.pt')
        with open(vocab_path, 'rb') as f:
            my_model.char_vocab = pickle.load(f)
        my_model.model.init_with_vocab(my_model.char_vocab)
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
