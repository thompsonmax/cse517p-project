import matplotlib.pyplot as plt
from datasets import load_dataset
import subprocess
import json
import pandas as pd
from wikimedia_languages import ALL_LANGUAGES
import numpy as np

#total_rows = 61614907
languages_used_as_list = ['en', 'de', 'fr', 'es', 'ru', 'ja', 'it', 'zh', 'pt', 'ar', 'pl', 'nl']

def get_languages():
    if len(ALL_LANGUAGES) != 0:
        language_lengths = ALL_LANGUAGES
        language_lengths = sorted(language_lengths, key=lambda x: x['rows'], reverse=True)
        return language_lengths

    result = subprocess.run(['curl', 'https://datasets-server.huggingface.co/splits?dataset=wikimedia%2Fwikipedia'], capture_output=True, text=True)
    response = json.loads(result.stdout)

    language_files = [ x['config'] for x in response['splits'] if not x['config'].endswith('.be-x-old') ] 
    language_files = list(set(language_files))

    language_lengths = []

    for file_name in language_files:
        print('processing:', file_name)
        dataset = load_dataset("wikimedia/wikipedia", file_name)
        df = pd.DataFrame(dataset['train'].to_pandas())
        last_period_index = file_name.rfind('.')
        language = file_name[last_period_index + 1:]
        
        language_lengths.append({
            'language': language,
            'rows': len(df)
        })

        #delete_results = !rm -r ~/.cache/huggingface/datasets

        del df
        del dataset
    
    language_lengths = sorted(language_lengths, key=lambda x: x['rows'], reverse=True)
    return language_lengths

if __name__ == "__main__":

    all_languages = get_languages()
    total_number_of_languages = len(all_languages)

    languages = [ l['language'] for l in all_languages]
    rows = [ l['rows'] for l in all_languages]

    total_rows = sum(rows)

    print('=== graph of language distribution ===')
    bins = np.logspace(1, np.log10(np.max(rows) + 10), num=20)
    plt.figure(figsize=(10, 6)) 
    plt.hist(rows, bins=bins, edgecolor='black')
    plt.xlabel('Number of Samples per Language')
    plt.xticks(rotation=45)
    plt.ylabel('Number of Languages')
    plt.xscale('log')
    plt.title('Language Distribution in Entire Dataset')
    plt.grid(axis='y', alpha=0.75)
    plt.show()

    print('=== graph of top n languages ===')
    n = 25
    plt.bar(languages[:n], rows[:n])
    plt.xlabel('Languages')
    plt.xticks(languages[:n], languages[:n], rotation=45)
    plt.ylabel('Samples')
    plt.yscale('log')
    plt.title(f'Top {n} Languages in Entire Dataset')
    plt.show()

    languages_used_dict = [ l for l in all_languages if l['language'] in languages_used_as_list]

    languages_used = [ l['language'] for l in languages_used_dict]
    rows_used = [ l['rows'] for l in languages_used_dict]
   
    print('=== graph of fetched languages ===')

    plt.bar(languages_used, rows_used)
    plt.xlabel('Languages')
    plt.xticks(languages_used, languages_used, rotation=45)
    plt.ylabel('Samples')
    plt.yscale('log')
    plt.title(f'Distribution of Fetched Languages')
    plt.show()

    print('=== summary of entire dataset ===')

    print('total samples:', total_rows)
    print('number of languages:', total_number_of_languages)

    dataset_stats = {
        'languages': total_number_of_languages,
        'mean': np.mean(rows),
        'median': np.median(rows),
        'max': np.max(rows),
        'min': np.min(rows),
        'std': np.std(rows),
        '25%': np.percentile(rows, 25),
        '50%': np.percentile(rows, 50),
        '75%': np.percentile(rows, 75),
        '90%': np.percentile(rows, 90),
        '95%': np.percentile(rows, 95),
        '99%': np.percentile(rows, 99)
    }

    print(f'average samples per language: {dataset_stats["mean"]:.2f}')
    print(f'median samples per language: {dataset_stats["median"]:.2f}')
    print(f'max samples for a language: {dataset_stats["max"]}')
    print(f'min samples for a language: {dataset_stats["min"]}')
    print(f'standard deviation: {dataset_stats["std"]:.2f}')
    print()
    print('# distribution')
    print(f'25th percentile: {dataset_stats["25%"]}')
    print(f'50th percentile: {dataset_stats["50%"]}')
    print(f'75th percentile: {dataset_stats["75%"]}')
    print(f'90th percentile: {dataset_stats["90%"]}')
    print(f'95th percentile: {dataset_stats["95%"]}')
    print(f'99th percentile: {dataset_stats["99%"]}')
    print()
    print('# top 5 languages')
    print(languages[:5])
    print()

    print('=== summary of used dataset ===')

    print('number of languages used:', len(languages_used))
    total_rows_used = sum(rows_used)
    percent_of_rows_used = total_rows_used / total_rows * 100
    print(f'total rows used: {total_rows_used} ({percent_of_rows_used:.2f}%)')

    dataset_used_stats = {
        'languages': len(languages_used),
        'mean': np.mean(rows_used),
        'median': np.median(rows_used),
        'max': np.max(rows_used),
        'min': np.min(rows_used),
        'std': np.std(rows_used),
        '25%': np.percentile(rows_used, 25),
        '50%': np.percentile(rows_used, 50),
        '75%': np.percentile(rows_used, 75),
        '90%': np.percentile(rows_used, 90),
        '95%': np.percentile(rows_used, 95),
        '99%': np.percentile(rows_used, 99)
    }

    print(f'average samples per language: {dataset_used_stats["mean"]:.2f}')
    print(f'median samples per language: {dataset_used_stats["median"]:.2f}')
    print(f'max samples for a language: {dataset_used_stats["max"]}')
    print(f'min samples for a language: {dataset_used_stats["min"]}')
    print(f'standard deviation: {dataset_used_stats["std"]:.2f}')
    print()
    print('# distribution')
    print(f'25th percentile: {dataset_used_stats["25%"]}')
    print(f'50th percentile: {dataset_used_stats["50%"]}')
    print(f'75th percentile: {dataset_used_stats["75%"]}')
    print(f'90th percentile: {dataset_used_stats["90%"]}')
    print(f'95th percentile: {dataset_used_stats["95%"]}')
    print(f'99th percentile: {dataset_used_stats["99%"]}')
    print()
    print('# top 5 languages')
    print(languages_used[:5])
    print()

    print('=== all the stats together ===')
    data = []
    for key in dataset_stats:
        data.append([key, f'{dataset_stats[key]:.2f}', f'{dataset_used_stats[key]:.2f}'])
    fig, ax = plt.subplots(figsize=(8, 4))
    table = ax.table(cellText=data, colLabels=['Statistic', 'Entire Dataset', 'Used Dataset'], cellLoc='center', loc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.axis('off')
    plt.title('Wikimedia Statistics')
    plt.show()
