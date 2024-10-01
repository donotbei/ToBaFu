import os

import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch import nn

from data.utils import read_data


def test_fusion_model(
        model1,
        model2,
        test_loader,
        loss=nn.CrossEntropyLoss(),
        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
        save_path=None,
        pool_method=None,
):
    model1.eval()
    model2.eval()

    # test the model
    test_loss = test_loss1 = test_loss2 = 0
    correct = correct1 = correct2 = 0
    with torch.no_grad():
        for data, target in test_loader:
            X1, X2 = data
            X1, X2, target = X1.to(device), X2.to(device), target.to(device)
            output1 = model1(X1)
            output2 = model2(X2)
            if pool_method == 'average':
                output = (output1 + output2) / 2
            elif pool_method == 'max':
                output = torch.max(output1, output2)
            test_loss += loss(output, target).item()
            test_loss1 += loss(output1, target).item()
            test_loss2 += loss(output2, target).item()
            pred = output.argmax(dim=1, keepdim=True)
            pred1 = output1.argmax(dim=1, keepdim=True)
            pred2 = output2.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            correct1 += pred1.eq(target.view_as(pred1)).sum().item()
            correct2 += pred2.eq(target.view_as(pred2)).sum().item()
    # get the confusion matrix
    conf_mat = confusion_matrix(target.cpu().numpy(), pred.cpu().numpy())
    conf_mat1 = confusion_matrix(target.cpu().numpy(), pred1.cpu().numpy())
    conf_mat2 = confusion_matrix(target.cpu().numpy(), pred2.cpu().numpy())

    # get the test loss
    data_size = len(test_loader.dataset)
    test_loss /= data_size
    test_loss1 /= data_size
    test_loss2 /= data_size

    # get the F1 score
    f1 = f1_score(target.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    f1_1 = f1_score(target.cpu().numpy(), pred1.cpu().numpy(), average='weighted')
    f1_2 = f1_score(target.cpu().numpy(), pred2.cpu().numpy(), average='weighted')

    # get the precision
    precision = precision_score(target.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    precision1 = precision_score(target.cpu().numpy(), pred1.cpu().numpy(), average='weighted')
    precision2 = precision_score(target.cpu().numpy(), pred2.cpu().numpy(), average='weighted')

    # get the recall
    recall = recall_score(target.cpu().numpy(), pred.cpu().numpy(), average='weighted')
    recall1 = recall_score(target.cpu().numpy(), pred1.cpu().numpy(), average='weighted')
    recall2 = recall_score(target.cpu().numpy(), pred2.cpu().numpy(), average='weighted')

    # print test sets: Confusion Matrix, F1 score, Precision, Recall
    print(f'Fusion Model: \n {conf_mat} \n F1: {f1:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}')
    print(f'TOPOmodel: \n {conf_mat1} \n F1: {f1_1:.4f}, Precision: {precision1:.4f}, Recall: {recall1:.4f}')
    print(f'IMAGEmodel: \n {conf_mat2} \n F1: {f1_2:.4f}, Precision: {precision2:.4f}, Recall: {recall2:.4f}')

    # print Three test set: Average loss and Accuracy
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, data_size,
        100. * correct / data_size))
    print('\nTOPOmodel Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss1, correct1, data_size,
        100. * correct1 / data_size))
    print('\nIMAGEmodel Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss2, correct2, data_size,
        100. * correct2 / data_size))

    # save the results
    with open(save_path, 'w') as f:
        f.write('Confusion Matrix: \n' + str(conf_mat) + '\n')
        f.write('F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n'.format(f1, precision, recall))
        f.write('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss, correct, data_size, 100. * correct / data_size))
        f.write('\nTOPOmodel Confusion Matrix: \n' + str(conf_mat1) + '\n')
        f.write('F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n'.format(f1_1, precision1, recall1))
        f.write('\nTOPOmodel Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss1, correct1, data_size, 100. * correct1 / data_size))
        f.write('\nIMAGEmodel Confusion Matrix: \n' + str(conf_mat2) + '\n')
        f.write('F1: {:.4f}, Precision: {:.4f}, Recall: {:.4f}\n'.format(f1_2, precision2, recall2))
        f.write('\nIMAGEmodel Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
            test_loss2, correct2, data_size, 100. * correct2 / data_size))


def test_fusion_models(config):
    # get the test dataloader
    combined_loader = read_data('test')

    # get the save_path
    TOPOmodel_path = os.path.join(config['model_path'], 'topo_checkpoint.pt')
    IMGmodel_path = os.path.join(config['model_path'], 'image_checkpoint.pt')

    # load the model
    TOPOmodel = torch.load(TOPOmodel_path)
    IMGmodel = torch.load(IMGmodel_path)

    # test the model
    for pool_method in ['average', 'max']:
        save_path = os.path.join(config['model_path'], pool_method+'fusion_test_results.txt')
        test_fusion_model(TOPOmodel, IMGmodel, combined_loader, save_path=save_path, pool_method=pool_method)


def test():
    config = OmegaConf.load('config/config.yaml')['test']
    test_fusion_models(config)


if __name__ == '__main__':
    test()
