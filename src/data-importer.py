#!/usr/bin/env python
from datasets import load_dataset
import pandas as pd


class DataImporter:
    """
    Class to fetch data  
    """

    @classmethod
    def load_common_corpus(self, data_files = None):
        # common corpus only contains split of 'train'
        dataset = load_dataset("PleIAs/common_corpus", data_files=data_files)
        df = pd.DataFrame(dataset['train'].to_pandas())
        print(f'Available columns: {list(df)}')
        df = df[df['language_type'] != 'Code'].reset_index(drop=True)
        return df
        

    @classmethod
    def divide_corpus_into_datasets(self, common_corpus, percent_dev=0.2, random_state=42):
        dev_dataset = common_corpus.sample(frac=percent_dev, random_state=random_state)
        train_dataset = common_corpus.drop(dev_dataset.index)
        return train_dataset, dev_dataset
    
# smaller dataset for testing
common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files="common_corpus_10/subset_100_1.parquet")
print(f'original corpus size: {common_corpus.shape}')
train_dataset, dev_dataset = DataImporter.divide_corpus_into_datasets(common_corpus)
print(f'training dataset size: {train_dataset.shape}')
print(f'dev dataset size: {dev_dataset.shape}')