import csv
import os
import numpy as np
import cv2

from gudhi.cubical_complex import CubicalComplex


def img_persistence(img_path):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    arr = np.array(img)
    persistence = []

    for j in range(c):
        channel = arr[:, :, c - j - 1]
        # calculate the persistent homology
        bcc = CubicalComplex(top_dimensional_cells=channel.flatten(), dimensions=[w, h])
        persistence.append(bcc.persistence())

    return persistence


def topo_feature(persistence):
    topo_feature = []

    for channel_persistence in persistence:
        n, n0, n1 = 0, 0, 0
        B0min, B0total = float('inf'), 0
        B1min, B1total, D1total = float('inf'), 0, 0
        P1max, P1total = float('-inf'), 0

        dim0_n = [0 for _ in range(10)]
        dim1_n = [0 for _ in range(10)]
        birth_intervals = [30, 60, 90, 120, 150, 180, 210, 240]
        for dim, k in channel_persistence:
            birth, death = k
            life = death - birth

            # number of 0, 1 dimensional features
            n += 1
            n0 += 1 if dim == 0 else 0
            n1 += 1 if dim == 1 else 0

            # sum of birth pixel values of 0, 1 dimensional features
            B0min = min(B0min, birth) if dim == 0 else B0min
            B0total += birth if dim == 0 else 0
            B1min = min(B1min, birth) if dim == 1 else B1min
            B1total += birth if dim == 1 else 0
            # sum of death pixel values of 1 dimensional features
            D1total += death if dim == 1 else 0

            # sum of life pixel values of 1 dimensional features
            P1max = max(P1max, life) if dim == 1 else P1max
            P1total += life if dim == 1 else 0

            # distribution of birth pixel values of 0, 1 dimensional features
            if dim == 0:
                for i, upper_limit in enumerate(birth_intervals):
                    if birth <= upper_limit:
                        dim0_n[i] += 1
                        break
                else:
                    dim0_n[8] += 1

                if life >= 10:
                    dim0_n[9] += 1

            if dim == 1:
                for i, upper_limit in enumerate(birth_intervals):
                    if birth <= upper_limit:
                        dim1_n[i] += 1
                        break
                else:
                    dim1_n[8] += 1

                if life >= 10:
                    dim1_n[9] += 1

        # mean of birth pixel values of 0, 1 dimensional features
        B0mean, B1mean = B0total / n0 if n0 > 0 else 0, B1total / n1 if n1 > 0 else 0
        # mean of death, life pixel values of 1 dimensional features
        D1mean = D1total / n1 if n1 > 0 else 0
        P1mean = P1total / n1 if n1 > 0 else 0
        # 30 topo features of single channel
        topo_feature += [n, n0, n1, B0min, B0mean, B1min, B1mean, D1mean, P1mean, P1max] + dim0_n + dim1_n

    return topo_feature


def topo_features_S(data_dir, save_path):
    image_categories = os.listdir(data_dir)
    image_categories.sort()
    if '.DS_Store' in image_categories:
        image_categories.remove('.DS_Store')

    topo_features = []
    for label, category in enumerate(image_categories):
        category_path = os.path.join('CRC_5000', category)
        image_files = os.listdir(category_path)
        if '.DS_Store' in image_files:
            image_files.remove('.DS_Store')
        image_files.sort()

        for image_file in image_files:
            # get the persistence of the image
            img_path = os.path.join(category_path, image_file)
            image_persistence = img_persistence(img_path)

            # get the topo features of the image
            image_path = os.path.join(category, image_file)
            topo_features.append([image_path] + topo_feature(image_persistence) + [label])
            print(f'Image {image_path} processed')

    with open(save_path, 'w', newline='') as csvfile:
        # dimension names of birth pixel values of 0, 1 dimensional features
        dim0_names = ['dim0_n' + str(i * 30) for i in range(1, 11)]
        dim0_names[-2], dim0_names[-1] = 'dim0_nlarger', 'dim0_P10'

        dim1_names = ['dim1_n' + str(i * 30) for i in range(1, 11)]
        dim1_names[-2], dim1_names[-1] = 'dim1_nlarger', 'dim1_P10'

        feature_names = ['n', 'n0', 'n1', 'B0min', 'B0mean', 'B1min', 'B1mean', 'D1mean', 'P1mean', 'P1max']
        feature_names += dim0_names + dim1_names

        writer = csv.writer(csvfile)
        feature_names = ['image_path'] + feature_names * 3 + ['label']
        writer.writerow(feature_names)

        for feature in topo_features:
            writer.writerow(feature)

    print(f'Topo features saved to {save_path}')
    return


if __name__ == '__main__':
    topo_features_S('../CRC_5000', 'CRC5000_S.csv')
    topo_features_S('../BUS250', 'BUS250_S.csv')
