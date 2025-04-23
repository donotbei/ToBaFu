import torch
import torch.nn as nn
from omegaconf import OmegaConf
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.optim import Adam
from data.utils import read_data
import os


config = OmegaConf.load('config/config.yaml')
#print("Config:\n", config)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_fusion_model(Topomodel,
                      ResNetmodel,
                      test_loader,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      save_path=None):

    Topomodel.eval()
    ResNetmodel.eval()

    best_accuracy = 0.0
    best_weight_Topo = 0.0
    best_weight_Img = 0.0
    best_results = {}
    weight_accuracy_map = {}
    weight_range = torch.linspace(0, 1, steps=21) 

    with torch.no_grad():
        for weight_Img in weight_range:
            weight_Topo = 1.0 - weight_Img

            correct_ToBaFu = 0
            correct_Topo = 0
            correct_Img = 0
            total = 0

            all_targets = []
            all_preds_ToBaFu = []
            all_preds_Topo = []
            all_preds_Img = []

            test_loss_ToBaFu = 0.0
            test_loss_Topo = 0.0
            test_loss_Img = 0.0


            for data, target in test_loader:
                X1, X2 = data
                X1, X2, target = X1.to(device), X2.to(device), target.to(device)
                output1 = Topomodel(X1)
                output2 = ResNetmodel(X2)

                output1 = F.softmax(output1, dim=1)
                output2 = F.softmax(output2, dim=1)


                ToBaFu_output = weight_Topo * output1 + weight_Img * output2


                loss_ToBaFu = F.cross_entropy(ToBaFu_output, target)
                loss_Topo = F.cross_entropy(output1, target)
                loss_Img = F.cross_entropy(output2, target)

                test_loss_ToBaFu += loss_ToBaFu.item()
                test_loss_Topo += loss_Topo.item()
                test_loss_Img += loss_Img.item()

                _, predicted_ToBaFu = torch.max(ToBaFu_output.data, 1)
                _, predicted_Topo = torch.max(output1.data, 1)
                _, predicted_Img = torch.max(output2.data, 1)

                total += target.size(0)
                correct_ToBaFu += (predicted_ToBaFu == target).sum().item()
                correct_Topo += (predicted_Topo == target).sum().item()
                correct_Img += (predicted_Img == target).sum().item()

                all_targets.extend(target.cpu().numpy())
                all_preds_ToBaFu.extend(predicted_ToBaFu.cpu().numpy())
                all_preds_Topo.extend(predicted_Topo.cpu().numpy())
                all_preds_Img.extend(predicted_Img.cpu().numpy())

            accuracy_ToBaFu = 100 * correct_ToBaFu / total
            accuracy_Topo = 100 * correct_Topo / total
            accuracy_Img = 100 * correct_Img / total

            weight_accuracy_map[weight_Topo.item()] = accuracy_ToBaFu

            if accuracy_ToBaFu > best_accuracy:
                best_accuracy = accuracy_ToBaFu
                best_weight_Topo = weight_Topo.item()
                best_weight_Img = weight_Img.item()

                best_results = {
                    #'conf_mat_ToBaFu': confusion_matrix(all_targets, all_preds_ToBaFu,labels=[1, 0]), #for the binary BUS250 dataset
                    'conf_mat_ToBaFu': confusion_matrix(all_targets, all_preds_ToBaFu), #for the multiple-class LC25000 or CRC5000 dataset
                    'f1_ToBaFu': f1_score(all_targets, all_preds_ToBaFu, average='weighted'),#delete "average='weighted'" for the binary BUS250 dataset
                    'precision_ToBaFu': precision_score(all_targets, all_preds_ToBaFu, average='weighted'),
                    'recall_ToBaFu': recall_score(all_targets, all_preds_ToBaFu, average='weighted'),
                    'test_loss_ToBaFu': test_loss_ToBaFu / total,
                    'accuracy_ToBaFu': accuracy_ToBaFu,

                    #'conf_mat_Topo': confusion_matrix(all_targets, all_preds_Topo,labels=[1, 0]), #for the BUS250 dataset
                    'conf_mat_Topo': confusion_matrix(all_targets, all_preds_Topo), #for the LC25000 or CRC5000 dataset
                    'f1_Topo': f1_score(all_targets, all_preds_Topo, average='weighted'),
                    'precision_Topo': precision_score(all_targets, all_preds_Topo, average='weighted'),
                    'recall_Topo': recall_score(all_targets, all_preds_Topo, average='weighted'),
                    'test_loss_Topo': test_loss_Topo / total,
                    'accuracy_Topo': accuracy_Topo,

                    #'conf_mat_Img': confusion_matrix(all_targets, all_preds_Img,labels=[1, 0]), #for the BUS250 dataset
                    'conf_mat_Img': confusion_matrix(all_targets, all_preds_Img), #for the LC25000 or CRC5000 dataset
                    'f1_Img': f1_score(all_targets, all_preds_Img, average='weighted'),
                    'precision_Img': precision_score(all_targets, all_preds_Img, average='weighted'),
                    'recall_Img': recall_score(all_targets, all_preds_Img, average='weighted'),
                    'test_loss_Img': test_loss_Img / total,
                    'accuracy_Img': accuracy_Img,
                }


    with open(save_path, 'w') as f:
        f.write(f'Best Test Accuracy: {best_accuracy:.2f}%\n')
        f.write(f'Weight_Topo: {best_weight_Topo:.2f}\n')
        f.write(f'Weight_ModifiedResNet: {best_weight_Img:.2f}\n')
        f.write(f'\nToBaFu Model: \nConfusion Matrix: \n{best_results["conf_mat_ToBaFu"]}\n')
        f.write(f'F1: {best_results["f1_ToBaFu"]:.4f}, Precision: {best_results["precision_ToBaFu"]:.4f}, Recall: {best_results["recall_ToBaFu"]:.4f}\n')
        f.write(f'Average loss: {best_results["test_loss_ToBaFu"]:.4f}, Accuracy: {best_results["accuracy_ToBaFu"]:.2f}%\n')

        f.write(f'\nTopo model: \nConfusion Matrix: \n{best_results["conf_mat_Topo"]}\n')
        f.write(f'F1: {best_results["f1_Topo"]:.4f}, Precision: {best_results["precision_Topo"]:.4f}, Recall: {best_results["recall_Topo"]:.4f}\n')
        f.write(f'Average loss: {best_results["test_loss_Topo"]:.4f}, Accuracy: {best_results["accuracy_Topo"]:.2f}%\n')

        f.write(f'\nModifiedResNet Model: \nConfusion Matrix: \n{best_results["conf_mat_Img"]}\n')
        f.write(f'F1: {best_results["f1_Img"]:.4f}, Precision: {best_results["precision_Img"]:.4f}, Recall: {best_results["recall_Img"]:.4f}\n')
        f.write(f'Average loss: {best_results["test_loss_Img"]:.4f}, Accuracy: {best_results["accuracy_Img"]:.2f}%\n')

        f.write('\nWeight and Accuracy:\n')
        for weight, accuracy in weight_accuracy_map.items():
            f.write(f'Weight_Topo: {weight:.2f},Weight_ModifiedResNet: {1-weight:.2f}, Accuracy: {accuracy:.2f}%\n')

    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    print(f'Weight_Topo: {best_weight_Topo:.2f}')
    print(f'Weight_ModifiedResNet: {best_weight_Img:.2f}\n')

    print(f'ToBaFu Model: \nConfusion Matrix: \n{best_results["conf_mat_ToBaFu"]}')
    print(f'F1: {best_results["f1_ToBaFu"]:.4f}, Precision: {best_results["precision_ToBaFu"]:.4f}, Recall: {best_results["recall_ToBaFu"]:.4f}')
    print(f'Average loss: {best_results["test_loss_ToBaFu"]:.4f}, Accuracy: {best_results["accuracy_ToBaFu"]:.2f}%\n')

    print(f'Topo model: \nConfusion Matrix: \n{best_results["conf_mat_Topo"]}')
    print(f'F1: {best_results["f1_Topo"]:.4f}, Precision: {best_results["precision_Topo"]:.4f}, Recall: {best_results["recall_Topo"]:.4f}')
    print(f'Average loss: {best_results["test_loss_Topo"]:.4f}, Accuracy: {best_results["accuracy_Topo"]:.2f}%\n')

    print(f'ModifiedResNet Model: \nConfusion Matrix: \n{best_results["conf_mat_Img"]}')
    print(f'F1: {best_results["f1_Img"]:.4f}, Precision: {best_results["precision_Img"]:.4f}, Recall: {best_results["recall_Img"]:.4f}')
    print(f'Average loss: {best_results["test_loss_Img"]:.4f}, Accuracy: {best_results["accuracy_Img"]:.2f}%\n')

    print('\nWeight and Accuracy:\n')

    for weight, accuracy in weight_accuracy_map.items():
        print(f'Weight_Topo: {weight:.2f},Weight_ModifiedResNet: {1-weight:.2f}, Accuracy: {accuracy:.2f}%\n')


def test_fusion_models(config_test):
    config_data = config['data']
    ToBaFu_loader = read_data(config_data, 'test')

    Topomodel_path = os.path.join(config_test['model_path'], 'topo_checkpoint.pt')
    ResNetmodel_path = os.path.join(config_test['model_path'], 'img_checkpoint.pt')

    Topomodel = torch.load(Topomodel_path, map_location=device)
    ResNetmodel = torch.load(ResNetmodel_path, map_location=device)
    save_path = os.path.join(config_test['model_path'], 'test_results.txt')

    test_fusion_model(Topomodel, ResNetmodel, ToBaFu_loader, save_path=save_path)

def test():
    config_test = config['test']
    test_fusion_models(config_test)

if __name__ == '__main__':
    test()