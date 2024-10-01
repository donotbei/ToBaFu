import os

import torch
from omegaconf import OmegaConf
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
from torch.nn import functional as F

from data.utils import read_data


def test_fusion_model(TOPOmodel,
                      ResNetmodel,
                      test_loader,
                      device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
                      save_path=None):

    TOPOmodel.eval()
    ResNetmodel.eval()

    best_accuracy = 0.0
    best_weight_topo = 0.0
    best_weight_img = 0.0
    best_results = {}

    weight_range = torch.linspace(0, 1, steps=21)

    with torch.no_grad():
        for weight_img in weight_range:
            weight_topo = 1.0 - weight_img

            correct_combined = 0
            correct_topo = 0
            correct_img = 0
            total = 0

            all_targets = []
            all_preds_combined = []
            all_preds_topo = []
            all_preds_img = []

            test_loss_combined = 0.0
            test_loss_topo = 0.0
            test_loss_img = 0.0

            for data, target in test_loader:
                X1, X2 = data
                X1, X2, target = X1.to(device), X2.to(device), target.to(device)
                output1 = TOPOmodel(X1)
                output2 = ResNetmodel(X2)

                output1 = F.softmax(output1, dim=1)
                output2 = F.softmax(output2, dim=1)

                combined_output = weight_topo * output1 + weight_img * output2

                loss_combined = F.cross_entropy(combined_output, target)
                loss_topo = F.cross_entropy(output1, target)
                loss_img = F.cross_entropy(output2, target)

                test_loss_combined += loss_combined.item()
                test_loss_topo += loss_topo.item()
                test_loss_img += loss_img.item()

                _, predicted_combined = torch.max(combined_output.data, 1)
                _, predicted_topo = torch.max(output1.data, 1)
                _, predicted_img = torch.max(output2.data, 1)

                total += target.size(0)
                correct_combined += (predicted_combined == target).sum().item()
                correct_topo += (predicted_topo == target).sum().item()
                correct_img += (predicted_img == target).sum().item()

                all_targets.extend(target.cpu().numpy())
                all_preds_combined.extend(predicted_combined.cpu().numpy())
                all_preds_topo.extend(predicted_topo.cpu().numpy())
                all_preds_img.extend(predicted_img.cpu().numpy())

            accuracy_combined = 100 * correct_combined / total
            accuracy_topo = 100 * correct_topo / total
            accuracy_img = 100 * correct_img / total

            if accuracy_combined > best_accuracy:
                best_accuracy = accuracy_combined
                best_weight_topo = weight_topo.item()
                best_weight_img = weight_img.item()

                best_results = {
                    'conf_mat_combined': confusion_matrix(all_targets, all_preds_combined),
                    'f1_combined': f1_score(all_targets, all_preds_combined, average='weighted'),
                    'precision_combined': precision_score(all_targets, all_preds_combined, average='weighted'),
                    'recall_combined': recall_score(all_targets, all_preds_combined, average='weighted'),
                    'test_loss_combined': test_loss_combined / total,
                    'accuracy_combined': accuracy_combined,

                    'conf_mat_topo': confusion_matrix(all_targets, all_preds_topo),
                    'f1_topo': f1_score(all_targets, all_preds_topo, average='weighted'),
                    'precision_topo': precision_score(all_targets, all_preds_topo, average='weighted'),
                    'recall_topo': recall_score(all_targets, all_preds_topo, average='weighted'),
                    'test_loss_topo': test_loss_topo / total,
                    'accuracy_topo': accuracy_topo,

                    'conf_mat_img': confusion_matrix(all_targets, all_preds_img),
                    'f1_img': f1_score(all_targets, all_preds_img, average='weighted'),
                    'precision_img': precision_score(all_targets, all_preds_img, average='weighted'),
                    'recall_img': recall_score(all_targets, all_preds_img, average='weighted'),
                    'test_loss_img': test_loss_img / total,
                    'accuracy_img': accuracy_img,
                }

    with open(save_path, 'w') as f:
        f.write(f'Best Test Accuracy: {best_accuracy:.2f}%\n')
        f.write(f'Weight_topo: {best_weight_topo:.2f}\n')
        f.write(f'Weight_img: {best_weight_img:.2f}\n')
        f.write(f'\nCombined Model: \nConfusion Matrix: \n{best_results["conf_mat_combined"]}\n')
        f.write(f'F1: {best_results["f1_combined"]:.4f}, Precision: {best_results["precision_combined"]:.4f}, Recall: {best_results["recall_combined"]:.4f}\n')
        f.write(f'Average loss: {best_results["test_loss_combined"]:.4f}, Accuracy: {best_results["accuracy_combined"]:.2f}%\n')

        f.write(f'\nTOPO model: \nConfusion Matrix: \n{best_results["conf_mat_topo"]}\n')
        f.write(f'F1: {best_results["f1_topo"]:.4f}, Precision: {best_results["precision_topo"]:.4f}, Recall: {best_results["recall_topo"]:.4f}\n')
        f.write(f'Average loss: {best_results["test_loss_topo"]:.4f}, Accuracy: {best_results["accuracy_topo"]:.2f}%\n')

        f.write(f'\nResNet model: \nConfusion Matrix: \n{best_results["conf_mat_img"]}\n')
        f.write(f'F1: {best_results["f1_img"]:.4f}, Precision: {best_results["precision_img"]:.4f}, Recall: {best_results["recall_img"]:.4f}\n')
        f.write(f'Average loss: {best_results["test_loss_img"]:.4f}, Accuracy: {best_results["accuracy_img"]:.2f}%\n')

    print(f'Best Test Accuracy: {best_accuracy:.2f}%')
    print(f'Weight_topo: {best_weight_topo:.2f}')
    print(f'Weight_img: {best_weight_img:.2f}\n')

    print(f'Combined Model: \nConfusion Matrix: \n{best_results["conf_mat_combined"]}')
    print(f'F1: {best_results["f1_combined"]:.4f}, Precision: {best_results["precision_combined"]:.4f}, Recall: {best_results["recall_combined"]:.4f}')
    print(f'Average loss: {best_results["test_loss_combined"]:.4f}, Accuracy: {best_results["accuracy_combined"]:.2f}%\n')

    print(f'TOPO model: \nConfusion Matrix: \n{best_results["conf_mat_topo"]}')
    print(f'F1: {best_results["f1_topo"]:.4f}, Precision: {best_results["precision_topo"]:.4f}, Recall: {best_results["recall_topo"]:.4f}')
    print(f'Average loss: {best_results["test_loss_topo"]:.4f}, Accuracy: {best_results["accuracy_topo"]:.2f}%\n')

    print(f'ResNet model: \nConfusion Matrix: \n{best_results["conf_mat_img"]}')
    print(f'F1: {best_results["f1_img"]:.4f}, Precision: {best_results["precision_img"]:.4f}, Recall: {best_results["recall_img"]:.4f}')
    print(f'Average loss: {best_results["test_loss_img"]:.4f}, Accuracy: {best_results["accuracy_img"]:.2f}%\n')


def test_fusion_models(config):
    config_data = OmegaConf.load('config/config.yaml')['data']
    combined_loader = read_data(config_data)

    TOPOmodel_path = os.path.join(config['model_path'], 'topo_checkpoint.pt')
    ResNetmodel_path = os.path.join(config['model_path'], 'image_checkpoint.pt')

    TOPOmodel = torch.load(TOPOmodel_path)
    ResNetmodel = torch.load(ResNetmodel_path)

    save_path = os.path.join(config['model_path'], 'test_results.txt')

    test_fusion_model(TOPOmodel, ResNetmodel, combined_loader, save_path=save_path)


def test():
    config = OmegaConf.load('config/config.yaml')['test']
    test_fusion_models(config)


if __name__ == '__main__':
    test()