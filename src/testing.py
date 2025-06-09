import pandas as pd
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter
from myprogram import MyModel
from pprint import pprint
import random
from datasets import load_dataset

UNICODE_BMP_MAX_CODE_POINT = 65535 # U+FFFF, spans Basic Multilingual Plane

def generate_test_data(data_files="papluca/language-identification", records=1000, random_state=42):
    dataset = load_dataset(data_files, split='train[0:10000]')
    df = dataset.to_pandas()
    validation_dataset = df.sample(n=records, random_state=random_state)
    samples = validation_dataset['text'].tolist()


    characters_to_skip = [ ' ', '\n', '.', ',', '(', ')' ]

    x = []
    y = []
    for sample in samples:
        if len(sample) < 5:
            continue
            
        length = random.choice(range(5, len(sample)-1))
        for i in range(0, 5):
            sample_length = length + i
            next_char = sample_length + 1
            if next_char >= len(sample) \
                or sample[next_char] in characters_to_skip \
                    or sample[sample_length] in characters_to_skip:
                continue
            
            input_text = sample[:sample_length + 1]
            expected_label = sample[next_char]

            x.append(input_text)
            y.append(expected_label)
            
    return x, y

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--work_dir', help='where to load the model from', default='work')
    parser.add_argument('--test_output', help='path to write entire output', default='work/testing_output.csv')
    args = parser.parse_args()

    print('Loading model')
    model = MyModel.load(args.work_dir)

    print('Generating test data')
    test_data, labels = generate_test_data()

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
