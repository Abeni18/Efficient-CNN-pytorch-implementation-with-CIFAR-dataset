import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR

import argparse
import os

import time
from tqdm import tqdm

from models.mobilenet_v2 import mobilenet_v2
from models.mobilenet_v2_2 import mobilenet_v2_2
from preprocess import load_data
# from plot import draw_plot
from torchviz import make_dot

use_cuda = torch.cuda.is_available()
# device = torch.device("cuda" if use_cuda else "cpu")

if use_cuda:
    device = torch.device("cuda")
    print('running on GPU')
else:
    device = torch.device('cpu')
    print('running on CPU')



"""
Adjust the learning rate / decay it at each 30 epochs

Possible improvement 

1. Faster training time - adjustable decay technique
2. 
"""


def adjust_learning_rate(optimizer, epoch, args):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.learning_rate * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



"""
Training the model 

Possible improvement 

1. Faster training time - adjustable decay technique
2. 
"""
def train(model, train_loader, optimizer, criterion, epoch, args):
    model.train()
    step = 0
    train_loss = 0
    train_acc = 0
    for data, target in tqdm(train_loader, desc="epoch " + str(epoch), mininterval=1):

        'Adjust the learning rate before anything started '
        adjust_learning_rate(optimizer, epoch, args)

        'Put the data and the label on GPU'
        data, target = data.to(device), target.to(device)   # loading data to the GPU device

        'Inizilize the optimizer to zero gradiant '
        optimizer.zero_grad()

        'Fit the data to the model and get the prediction output'
        output = model(data)

        g = make_dot(output.mean(), params=dict(model.named_parameters()))
        g.format = 'svg'
        g.format = 'pdf'
        g.filename = 'image_main_skip_orginal_image'
        g.render(view=False)

        'Calculate the loss'
        loss = criterion(output, target)

        'Backpropagate the loss'
        loss.backward()

        'Optimize the weights and biases'
        optimizer.step()

        'Save the traning loss in the training_loss list'
        train_loss += loss.data

        'Convert the one hot encoded output to the integer '
        y_pred = output.data.max(1)[1]

        'Calculate the loss by comparing the y_predicted to y_label'
        acc = float(y_pred.eq(target.data).sum()) / len(data) * 100.

        'Save the training accuracy in train_acc list'
        train_acc += acc

        'Initialize step for printing out the results'
        step += 1

        'After every 100 iteration print out the results'
        if step % 100 == 0:
            print("[Epoch {0:4d}] Loss: {1:2.3f} Acc: {2:.3f}%".format(epoch, loss.data, acc), end='')
            for param_group in optimizer.param_groups:
                print(",  Current learning rate is: {}".format(param_group['lr']))


    ' Return the average training loss and training accuracy '
    length = len(train_loader.dataset) // args.batch_size
    return train_loss / length, train_acc / length



'Calculate test accuracy'
def get_test(model, test_loader):

    model.eval()
    'model.eval() - notify all your layers you are ino eval mode that way, batchnorm or dropout layers will work in eval mode instead of training mode'
    correct = 0
    'torch.no_grad() - impacts the autograd engine and deactivate it. It will reduce memory usage and speed up computations but you won’t be able to backprop (which you don’t want in an eval script).'
    with torch.no_grad():
        for data, target in tqdm(test_loader, desc="evaluation", mininterval=1):
            data, target = data.to(device), target.to(device)  # load the data to device (gpu)
            output = model(data)                               # get the output from the model (notice that the model is setup to be in prediction mode)
            prediction = output.data.max(1)[1]                 # get the maximum value out of the predicted probablities (i guess this is softmax)
            correct += prediction.eq(target.data).sum()        # compare the pridction with the target (label)

    acc = 100. * float(correct) / len(test_loader.dataset)     # calculate the accuracy in percent
    return acc


def main():
    parser = argparse.ArgumentParser('parameters')

    'Graph Hyper parameters '
    # parser.add_argument('--p', type=float, default=0.75, help='graph probability, (default: 0.75)')
    # parser.add_argument('--c', type=int, default=109, help='channel count for each node, (example: 78, 109, 154), (default: 154)')
    # parser.add_argument('--k', type=int, default=16, help='each node is connected to k nearest neighbors in ring topology, (default: 4)')
    # parser.add_argument('--m', type=int, default=15, help='number of edges to attach from a new node to existing nodes, (default: 5)')
    # parser.add_argument('--graph-mode', type=str, default="WS", help="random graph, (Example: ER, WS, BA), (default: WS)")
    # parser.add_argument('--node-num', type=int, default=32, help="Number of graph node (default n=32)")

    'Typical DNN hyper parameters'
    parser.add_argument('--epochs', type=int, default=500, help='number of epochs, (default: 100)')
    parser.add_argument('--learning-rate', type=float, default=1e-1, help='learning rate, (default: 1e-1)')
    parser.add_argument('--batch-size', type=int, default=5000, help='batch size, (default: 100)')
    # parser.add_argument('--model-mode', type=str, default="CIFAR10", help='CIFAR10, CIFAR100, SMALL_REGIME, REGULAR_REGIME, (default: CIFAR10)')
    parser.add_argument('--dataset-mode', type=str, default="CIFAR10", help='Which dataset to use? (Example, CIFAR10, CIFAR100, MNIST, Imagenet), (default: CIFAR10)')
    parser.add_argument('--is-train', type=bool, default=True, help="True if training, False if test. (default: True)")
    parser.add_argument('--load-model', type=bool, default=False)

    args = parser.parse_args()

    'Use the predefined args to get the dataset type and to preprocess the data and load it for training and test (loading data for different dataset is different)'
    train_loader, test_loader = load_data(args)


    'Initializing the model.py using args parameters to variable - model'

    'If load-model is true - after initializing the model it saves it and print out'
    if args.load_model:
        # model = Model(args.node_num, args.p, args.c, args.c, args.graph_mode, args.model_mode, args.dataset_mode, args.is_train).to(device)
        # model = mobilenet_v2(True).to(device)

        model = mobilenet_v2_2().to(device)
        filename = "mobilenet_v2"+ "_dataset_" + args.dataset_mode
        checkpoint = torch.load('./checkpoint/' + filename + 'ckpt.t7')
        model.load_state_dict(checkpoint['model'])
        epoch = checkpoint['epoch']
        acc = checkpoint['acc']
        print("Load Model Accuracy: ", acc, "Load Model end epoch: ", epoch)

        ' Else it only initialize the model'
    else:
        # model = mobilenet_v2(args.dataset_mode, pretrained=False).to(device)
        model = mobilenet_v2_2(args.dataset_mode).to(device)


    if device is 'cuda':

        'put the model on GPU and run the operation on multiple GPUS by making the omdel run parallelly '
        model = torch.nn.DataParallel(model)


    'Selecting schoastic gradient decent optimization algorithm (https://pytorch.org/docs/stable/optim.html)'
    optimizer = optim.SGD(model.parameters(), lr=args.learning_rate, weight_decay=5e-4, momentum=0.9)

    'Selecting loss function '
    criterion = nn.CrossEntropyLoss().to(device)

    epoch_list = []
    test_acc_list = []
    train_acc_list = []
    train_loss_list = []
    max_test_acc = 0
    if not os.path.isdir("reporting"):
        os.mkdir("reporting")

    start_time = time.time()
    with open("./reporting/" + "_dataset_" + args.dataset_mode + ".txt", "w") as f:
        for epoch in range(1, args.epochs + 1):
            # scheduler = CosineAnnealingLR(optimizer, epoch)
            epoch_list.append(epoch)

            'start training the model by passing - model, pre-processed data, optimizer ...'
            train_loss, train_acc = train(model, train_loader, optimizer, criterion, epoch, args)

            'After training 1 epoch do test accuracy calculation by passing , the trained model and pre-processed test data'
            test_acc = get_test(model, test_loader)

            test_acc_list.append(test_acc)
            train_loss_list.append(train_loss)
            train_acc_list.append(train_acc)
            print('Test set accuracy: {0:.2f}%, Best accuracy: {1:.2f}%'.format(test_acc, max_test_acc))
            f.write("[Epoch {0:3d}] Test set accuracy: {1:.3f}%, , Best accuracy: {2:.2f}%".format(epoch, test_acc, max_test_acc))
            f.write("\n ")

            if max_test_acc < test_acc:
                print('Saving..')
                state = {
                    'model': model.state_dict(),
                    'acc': test_acc,
                    'epoch': epoch,
                }
                if not os.path.isdir('checkpoint'):
                    os.mkdir('checkpoint')
                filename = "mobilenet_v2" + "_dataset_" + args.dataset_mode
                torch.save(state, './checkpoint/' + filename + 'ckpt.t7')
                max_test_acc = test_acc
                # draw_plot(epoch_list, train_loss_list, train_acc_list, test_acc_list)
            print("Training time: ", time.time() - start_time)
            f.write("Training time: " + str(time.time() - start_time))
            f.write("\n")


if __name__ == '__main__':
    main()
