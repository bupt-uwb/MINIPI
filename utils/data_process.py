import scipy.io as sio
import torch


def one_hot(num_classes, sort):
    oh_vector = []
    for i in range(num_classes):
        if i == sort - 1:
            oh_vector.append(1)
        else:
            oh_vector.append(0)
    return oh_vector


def pre_data(data_root, num_classes):
    dataset = sio.loadmat(data_root)
    train_dataset = torch.from_numpy(dataset['train'])[:, :100]\
        .reshape(len(dataset['train']), 1, 20, 5).to(torch.float32)
    test_dataset = torch.from_numpy(dataset['test'])[:, :100]\
        .reshape(len(dataset['test']), 1, 20, 5).to(torch.float32)
    train_gt = dataset['label_train'].tolist()
    test_gt = dataset['label_test'].tolist()
    for i in range(len(train_gt)):
        train_gt[i] = one_hot(num_classes, train_gt[i][0])
    for i in range(len(test_gt)):
        test_gt[i] = one_hot(num_classes, test_gt[i][0])
    train_gt = torch.tensor(train_gt).reshape(len(train_gt), 1, 1, 8).to(torch.float32)
    test_gt = torch.tensor(test_gt).reshape(len(test_gt), 1, 1, 8).to(torch.float32)
    # train_gt = train_gt.clone().detach().reshape(len(train_gt), 1, 1, 8).to(torch.float32)
    # test_gt = test_gt.clone().detach().reshape(len(test_gt), 1, 1, 8).to(torch.float32)
    return train_dataset, train_gt, test_dataset, test_gt

if __name__ == '__main__':
    dataset = sio.loadmat('../data/id_0514.mat')
    train_gt = dataset['label_train'].tolist()
    for i in range(len(train_gt)):
        train_gt[i] = one_hot(8, train_gt[i][0])
    train_gt = torch.tensor(train_gt).reshape(len(train_gt), 1, 1, 8).to(torch.float32)
    print(len(train_gt))
