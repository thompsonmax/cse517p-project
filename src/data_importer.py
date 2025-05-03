#!/usr/bin/env python
from datasets import load_dataset
import pandas as pd
from sklearn.model_selection import train_test_split


class DataImporter:
    """
    Class to fetch data  
    """

    @classmethod
    def load_common_corpus(self, data_files = None, verbose=True):
        # common corpus only contains split of 'train'
        dataset = load_dataset("PleIAs/common_corpus", data_files=data_files)
        df = pd.DataFrame(dataset['train'].to_pandas())
        if verbose:
            print(f'Available columns: {list(df)}')
        df = df[df['language_type'] != 'Code'].reset_index(drop=True)
        return df
    
    @classmethod
    def describe_dataset(self, common_corpus):
        language_counts = common_corpus['language'].value_counts()
        print('=== summary ===')
        print(f'total samples: {len(common_corpus)}')
        print(f'total unique languages: {len(language_counts)}')
        print(f'average samples per language: {language_counts.mean():.2f}')
        print(f'median samples per language: {language_counts.median():.2f}')
        print(f'max samples for a language: {max(language_counts)}')
        print(f'min samples for a language: {min(language_counts)}')
        print(f'standard deviation: {language_counts.std():.2f}')
        print()
        print('=== distribution ===')
        for percentile in [25, 50, 75, 90, 95, 99]:
            print(f'{percentile}th percentile samples: {language_counts.quantile(percentile / 100): 0.1f}')
        print()
        print('=== top 5 languages ===')
        print(language_counts.head(5))
        print()

    
    @classmethod 
    def sample_across_languages(self, common_corpus, minimum_samples=2, sample_size=None, random_state=42, verbose=True):
        language_counts = common_corpus['language'].value_counts()
        languages_to_sample = language_counts[language_counts >= minimum_samples].index.tolist()
        if verbose:
            print(f'{len(language_counts)} languages in original dataset, sampling {len(languages_to_sample)}')

        language_data = []
        for language in languages_to_sample:
            language_df = common_corpus[common_corpus['language'] == language]
            if sample_size is not None:
                language_df = language_df.sample(n=sample_size, random_state=random_state)
            language_data.append(language_df)
        
        stratified_df = pd.concat(language_data, ignore_index=True)
        return stratified_df

    @classmethod
    def divide_corpus_into_stratified_datasets(self, common_corpus, stratify=True, percent_dev=0.2, random_state=42):
        train_dataset, dev_dataset = train_test_split(
            common_corpus, 
            test_size=percent_dev, 
            stratify=(common_corpus['language'] if stratify is True else None), 
            random_state=random_state)
        return train_dataset, dev_dataset
    
if __name__ == "__main__":
    # smaller dataset for testing
    common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files="common_corpus_10/subset_100_*.parquet")
    print(f'\noriginal corpus size: {common_corpus.shape}\n')
    DataImporter.describe_dataset(common_corpus)

    print(f'=== with stratifying ===')
    common_corpus_stratified = DataImporter.sample_across_languages(common_corpus, minimum_samples=50, sample_size=50)
    print(f'stratified by language corpus size: {common_corpus_stratified.shape}')
    train_dataset, dev_dataset = DataImporter.divide_corpus_into_stratified_datasets(common_corpus_stratified)
    print(f'training dataset size: {train_dataset.shape}')
    print(f'dev dataset size: {dev_dataset.shape}')
    print()

    print(f'=== without stratifying ===')
    common_corpus_stratified = DataImporter.sample_across_languages(common_corpus, minimum_samples=1)
    print(f'stratified by language corpus size: {common_corpus_stratified.shape}')
    train_dataset, dev_dataset = DataImporter.divide_corpus_into_stratified_datasets(common_corpus_stratified, stratify=False)
    print(f'training dataset size without stratifying: {train_dataset.shape}')
    print(f'dev dataset size without stratifying: {dev_dataset.shape}')