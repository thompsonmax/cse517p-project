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

model_results = [
    {
        'epoch': 1,
        'training_loss': 1.8181,
        'dev_loss': 1.4443,
        'learning_rate': 0.0002,
    },
    {
        'epoch': 2,
        'training_loss': 1.3708,
        'dev_loss': 1.2293,
        'learning_rate': 0.00016,
    },
    {
        'epoch': 3,
        'training_loss': 1.2747,
        'dev_loss': 1.1955,
        'learning_rate': 0.000128,
    },
    {
        'epoch': 4,
        'training_loss': 1.2192,
        'dev_loss': 1.1558,
        'learning_rate': 8.192e-05,
    },
]

if __name__ == "__main__":
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    parser.add_argument('--performance-plot', type=bool, default=False, help='whether to plot model performance')
    parser.add_argument('--loss-plot', type=bool, default=True, help='whether to plot model performance')
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
    
    if args.loss_plot:
        epochs = [x['epoch'] for x in model_results]
        training_loss = [x['training_loss'] for x in model_results]
        dev_loss = [x['dev_loss'] for x in model_results]
        learning_rate = [x['learning_rate'] for x in model_results]
        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.plot(epochs, training_loss, label='Training Loss', marker='o')
        ax1.plot(epochs, dev_loss, label='Dev Loss', marker='o')
        ax1.set_xlabel('Epochs')
        ax1.set_ylabel('Loss')
        ax1.legend(loc='upper left', bbox_to_anchor=(-0.1, 1.1))
        ax1.grid(True, alpha=0.75)

        ax2 = ax1.twinx()
        ax2.plot(epochs, learning_rate, label='Learning Rate', marker='o', color='orange')
        ax2.set_ylabel('Learning Rate')
        ax2.legend(loc='upper right', bbox_to_anchor=(1.1, 1.1))
        ax2.grid(False)

        plt.title('Model Loss vs Learning Rate')
        plt.show()