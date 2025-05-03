import torch.nn as nn
import torch

class FFNN(nn.Module): # We inherit from nn.Module, which is the base class for all PyTorch Neural Network modules

    def __init__(
        self, input_dim: int, hidden_dim: int, num_classes: int, num_layers=3
    ):

        """
        Define the architecture of a Feedforward Neural Network with architecture described above.

        Inputs:
        - input_dim: The dimension of the input (d according to the figure above)
        - hidden_dim: The dimension of the hidden layer (h according to the figure above)
        - num_classes: The number of classes in the classification task.
        - num_layers: The number of hidden layers in the network.

        """

        super(FFNN, self).__init__() # Call the base class constructor

        self.num_layers = num_layers
        # Define your network architecture below

        self.fc1 = None # First linear layer
        self.act1 = None # First activation function
        self.fc2 = None # Second linear layer

        # YOUR CODE HERE
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.act1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, num_classes if num_layers == 1 else hidden_dim)

        if num_layers > 1:
            self.act2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_dim, num_classes if num_layers == 2 else hidden_dim)
        if num_layers > 2:
            self.act3 = nn.ReLU()
            self.fc4 = nn.Linear(hidden_dim, num_classes if num_layers == 3 else hidden_dim)
        if num_layers > 3:
            self.act4 = nn.ReLU()
            self.fc5 = nn.Linear(hidden_dim, num_classes)

        self.initialize_weights() # Initialize the weights of the linear layers

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        """
        Computes the forward pass through the network.

        Inputs:
        - x : Input tensor of shape (n, d) where n is the number of samples and d is the dimension of the input

        Hint: You can call a layer directly with the input to get the output tensor, e.g. self.fc1(x) will return the output tensor after applying the first linear layer.
        """

        # YOUR CODE HERE
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)

        if self.num_layers > 1:
            x = self.act2(x)
            x = self.fc3(x)
        if self.num_layers > 2:
            x = self.act3(x)
            x = self.fc4(x)
        if self.num_layers > 3:
            x = self.act4(x)
            x = self.fc5(x)

        return x

    def initialize_weights(self):
        """
        Initialize the weights of the linear layers.

        We initialize the weights using Xavier Normal initialization and the biases to zero.

        You can read more about Xavier Initialization here: https://cs230.stanford.edu/section/4/#xavier-initialization
        """

        for layer in self.children():
            if type(layer) == nn.Linear:
                nn.init.xavier_normal_(layer.weight)
                nn.init.zeros_(layer.bias)
