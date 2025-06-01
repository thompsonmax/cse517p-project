import matplotlib.pyplot as plt
from argparse import ArgumentParser, ArgumentDefaultsHelpFormatter

model_perfomance = [
    {
        'label': 'sentence transformer embeddings',
        'accuracy': 0.31,
    },
    {
        'label': 'checkpoint 2',
        'accuracy': 0.43,
    },
    {
        'label': 'move to decoder only model',
        'accuracy': 0.5,
    },
    {
        'label': 'use wikimedia + adamw optimizer',
        'accuracy': 0.42,
    },
]

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--performance-plot', type=bool, default=True, help='whether to plot model performance')
    args = parser.parse_args()

    if args.performance_plot:
        accuracies = [x['accuracy'] for x in model_perfomance]

        plt.figure(figsize=(10, 6))
        plt.plot(accuracies, marker='o')
        plt.xticks([])
        plt.ylabel('Accuracy')
        plt.title('Model Performance Over Time')
        plt.grid(True, alpha=0.75)
        plt.show()