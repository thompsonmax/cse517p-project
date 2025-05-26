import hyperparams
from data_importer import DataImporter
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

ENTIRE_CORPUS_SIZE=469938143

if __name__ == "__main__":

    length = []
    common_corpus: pd.DataFrame = DataImporter.load_common_corpus(data_files="common_corpus_10/subset_10*")

    common_corpus_stratified = DataImporter.sample_across_languages(common_corpus, minimum_samples=4, max_samples=hyperparams.DATASET_MAX_SAMPLES)
    train_dataset, dev_dataset = DataImporter.divide_corpus_into_datasets(common_corpus_stratified)
    x = train_dataset['text'].tolist()
    size = len(x)
    
    print(f'final dataset size: {size}, corpus size: {common_corpus.shape[0]}, entire dataset size: {ENTIRE_CORPUS_SIZE}')
    print(f'percent of data pulled initially: {common_corpus.shape[0] / ENTIRE_CORPUS_SIZE * 100:.2f}%')
    print(f'percent of data actually used for training: {size / ENTIRE_CORPUS_SIZE * 100:.2f}%')

    language_counts = common_corpus['language'].value_counts()
    language_counts_df = pd.DataFrame({'language': language_counts.index, 'count': language_counts.values})

    bins = np.logspace(0, np.log10(language_counts.max() + 10), num=20)
    plt.figure(figsize=(10, 6)) 
    plt.hist(language_counts.values, bins=bins, edgecolor='black')
    plt.xlabel('Number of Samples per Language')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Languages')
    plt.xscale('log')
    plt.title('Language Distribution in Fetched Dataset')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    top_25_languages = language_counts.nlargest(25)
    plt.figure(figsize=(10, 6))
    plt.bar(top_25_languages.index, top_25_languages.values)
    plt.xlabel('Language')
    plt.xticks(rotation=45)
    plt.ylabel('Sample Count')
    plt.yscale('log')
    plt.title('Top 25 Languages in Fetched Dataset')
    plt.grid(axis='y', alpha=0.75)
    plt.show()




