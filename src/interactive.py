import pandas as pd
from data_importer import DataImporter
import dataloader
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from myprogram import MyModel
from pprint import pprint
import torch
import random
import string
import hyperparams
import time

UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane

def progressive_print(text, delay=0.1):
  """
  Prints the characters of a string one at a time with a specified delay.

  Args:
    text: The string to be printed.
    delay: The time in seconds to wait between each character.
  """
  for char in text:
    print(char, end='', flush=True)
    time.sleep(delay)

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to load the model from', default='work')
    parser.add_argument('--input_seq', help='input sequence to repeatedly prompt for a new char', default='The key to the future is ')
    parser.add_argument('--random_seed', help='random seed for reproducibility', type=int, default=12471212)
    # parser.add_argument('--only_characters', help='remove punctuation and numbers from the test dataset', default=False, action='store_true')
    args = parser.parse_args()

    random.seed(args.random_seed)

    print('Loading model')
    model = MyModel.load(args.work_dir)

    curr_seq = args.input_seq

    print('Making predictions')
    for i in range(80):
        test_data = [curr_seq]  # Use the last SEQ_LENGTH characters of the current sequence
        pred = model.run_pred(test_data, verbose=True, filter_special_chars=True)
        # Sample the next character from the prediction.
        # Use probability of 0.5 for the first char, 0.3 for the second, and 0.2 for the third.
        probabilities = [0.80, 0.15, 0.05]
        next_char = random.choices(pred[0], weights=probabilities, k=1)[0]

        print(f'Predicted next char: {next_char}')
        curr_seq += next_char
        print(f'Current sequence: {curr_seq}')

    progressive_print(curr_seq, delay=0.05)