# need torch
import torch
import torch.nn as nn
import torch.nn.functional as F
from analysis import *
from data import *
from torch.utils.data import DataLoader, BatchSampler

def calc_accuracy(output, target):
    """Computes the accuracy for multiple binary predictions"""
    pred = output >= 0.5
    truth = target >= 0.5
    acc = float(pred.eq(truth).sum()) / float(target.numel())
    return acc/10

##############################
# MODEL                      #
##############################

# our class must extend nn.Module
class LanguageClassifier(nn.Module):
    def __init__(self,
                 input_size,
                 output_size):
        super(LanguageClassifier, self).__init__()

        # input
        self.fc1 = nn.Linear(input_size, 50)

        self.fc2 = nn.Linear(50, 10)

        # This applies linear transformation to produce output data
        self.fc3 = nn.Linear(10, output_size)

        # activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    # This must be implemented
    def forward(self, x):
        # Output of the first layer
        x = self.relu(self.fc1(x))

        x = self.relu(self.fc2(x))

        # This produces output
        output = self.sigmoid(self.fc3(x))
        return output


##############################
# Initialize                 #
##############################

#Initialize the model
model = LanguageClassifier(21,1)

#Define loss criterion
criterion = nn.BCELoss()

#Define the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


##############################
# Load data                  #
##############################

# load languages
train, generalize = load_metrics("runs/lstm_max_len_5_vocab_5_same_data_attr_5_split_2")

# todo extract seed?

# how are we going to compare different languages?
data = ClassifierDataset(train)
batch_size = 10

train_data = DataLoader(
                data,
                pin_memory=True,
                batch_sampler=BatchSampler(
                    ClassifierSampler(data, shuffle=True),
                    batch_size=batch_size,
                    drop_last=False,
                ),
            )

##############################
# TRAIN                      #
##############################

# Number of epochs
epochs = 500

for epoch in range(epochs):

    losses = []
    for (target, sample) in train_data:

        # todo improve target conversion
        target = target - 1

        pred = model(sample.float())

        loss = criterion(pred, target.float())
        losses.append(loss)

        # Clear the previous gradients
        optimizer.zero_grad()

        # Compute gradients
        loss.backward()

        # Adjust weights
        optimizer.step()

    if epoch % 10 == 0:
        print(pred)
        print(torch.mean(loss))


##############################
# TEST                       #
##############################

# how are we going to compare different languages?
data = ClassifierDataset(generalize)
batch_size = 10

test_data = DataLoader(
                data,
                pin_memory=True,
                batch_sampler=BatchSampler(
                    ClassifierSampler(data, shuffle=False),
                    batch_size=batch_size,
                    drop_last=False,
                ),
            )

print()
# loop through the test data
all_acc = []
for i, (target, sample) in enumerate(test_data):

    # todo improve target conversion
    target = target - 1

    # get test prediction
    pred = model(sample.float())

    # calculate accuracy
    acc = calc_accuracy(pred, target)
    all_acc.append(acc)
    print()
    print('Accuracy for batch {}: '.format(i), acc)

print()
print('Overall accuracy: ', np.mean(all_acc))
