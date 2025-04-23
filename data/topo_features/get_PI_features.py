# computing Persistence image and get PI features
import csv
from collections import defaultdict
import cv2
import numpy as np
import os
from gudhi.cubical_complex import CubicalComplex
from scipy.stats import multivariate_normal


def img_suffixes(img_path):
    img = cv2.imread(img_path)
    h, w, c = img.shape
    arr = np.array(img)
    persistence = []

    for j in range(c):
        channel = arr[:, :, c - j - 1]
        # calculate the persistent homology
        bcc = CubicalComplex(top_dimensional_cells=channel.flatten(), dimensions=[w, h])

        persistence.append(bcc.persistence())
    suffixes = []

    for channel_persistence in persistence:
        channel_suffixes = defaultdict(list)
        channel_suffixes_0 = []
        channel_suffixes_1 = []
        for i, (dim, (birth, death)) in enumerate(channel_persistence):
            if death - birth < 10:
                continue
            if death - birth == float('inf'):
                continue
            if dim == 0:
                channel_suffixes_0.append((birth, death - birth))
            if dim == 1:
                channel_suffixes_1.append((birth, death - birth))
        channel_suffixes[0] = channel_suffixes_0
        channel_suffixes[1] = channel_suffixes_1

        suffixes.append(channel_suffixes)

    return suffixes


def Persistence_surface(suffixes_dim):

    if not suffixes_dim:  # If the list is empty
        print("suffixes_dim is empty. Returning a default zero array.")
        return np.zeros((1, 3))  # return a default zero array


    suffixes_dim = np.array(suffixes_dim)

    max_second_dim = np.max(suffixes_dim[:, 1])
    n = len(suffixes_dim)
    weighted_sum = np.zeros(n)
    weighted_sums = []

    for i in range(n):
        mean = suffixes_dim[i]  # The i-th point in the data array
        cov = [[1, 0], [0, 1]]  # Covariance matrix with variances 1
        mvn = multivariate_normal(mean=mean, cov=cov)
        pdf_value = mvn.pdf(suffixes_dim)  # Calculate the joint density function pdf of the normal distribution at all points in data,
        weight = (mean[1] / max_second_dim)  # Calculate the weight
        weighted_pdf = pdf_value * weight  # Calculate the weighted joint density function
        weighted_sum += weighted_pdf  # Accumulate weighted_pdf into weighted_sum
        weighted_sums.append(weighted_sum.copy())  # Append the current weighted_sum to weighted_sums


    r = np.zeros((n, 3))
    r[:, 0:2] = suffixes_dim
    r[:, 2] = weighted_sum if n > 0 else np.zeros(n)

    return r

def Persistence_image(suffixes_r):
    x = suffixes_r[:, 0] if len(suffixes_r) > 0 else np.zeros(0)
    y = suffixes_r[:, 1] if len(suffixes_r) > 0 else np.zeros(0)
    z = suffixes_r[:, 2] if len(suffixes_r) > 0 else np.zeros(0)
    data = np.stack((x, y, z), axis=1)
    # Bin the x and y coordinates into intervals [0, 5] (not [0, 255] due to value range of Birth/Persistence)
    x_bins = np.linspace(0, 250, 6)
    y_bins = np.linspace(0, 250, 6)
    # Compute the interval indices for x and y, with bins defaulting to False, left-closed, right-open
    x_indices = np.digitize(x, x_bins, right=False) - 1
    y_indices = np.digitize(y, y_bins, right=False) - 1
    # Make sure the indexes are within the range of 0 to 4
    x_indices = np.clip(x_indices, 0, 4)
    y_indices = np.clip(y_indices, 0, 4)

    # Create a 5x5 array to store the sum of z coordinates within each region
    grid_sum = np.zeros((5, 5))
    # Iterate over the data and compute the sum of z coordinates within each region
    for k in range(len(data)):
        x_index = x_indices[k]
        y_index = y_indices[k]
        z_value = data[k, 2]  # Third column is the z coordinate
        grid_sum[y_index, x_index] += z_value
    flattened_sum = grid_sum.flatten()

    return list(flattened_sum)


def topo_features_PI(data_dir, save_path):
    data_dir_last = data_dir.split('/')[-1]
    #Set the value of start based on the last part of the data_dir
    if data_dir_last == 'BUS250':
        start = 0
    else:
        start = 1
    image_categories = os.listdir(data_dir)
    image_categories.sort()
    #img_categories = sorted(image_categories, key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

    PI_features = []

    for label, category in enumerate(image_categories,start=start): #label start form 0 or 1
        category_path = os.path.join(data_dir, category)
        image_files = os.listdir(category_path)
        # image_files.sort()
        image_files = sorted(image_files,
                             key=lambda x: int(''.join(filter(str.isdigit, x))) if any(c.isdigit() for c in x) else x)

        for image_file in image_files:
            PI_feature = []
            img_path = os.path.join(category_path, image_file)
            suffixes = img_suffixes(img_path)
            channel_1, channel_2, channel_3 = suffixes

            for channel in [channel_1, channel_2, channel_3]:
                PI_feature += Persistence_image(Persistence_surface(channel[0])) + \
                              Persistence_image(Persistence_surface(channel[1]))
            short_image_path = os.path.join(category, image_file)
            PI_features.append([short_image_path] + PI_feature + [label])

    with open(save_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)

        feature_names = ['image_path']
        for channel in range(1, 4):
            for dim in range(2):
                for j in range(1, 26):
                    feature_names.append(f'C{channel}_dim{dim}_{j}')
        feature_names.append('label')
        writer.writerow(feature_names)

        for feature in PI_features:
            writer.writerow(feature)

    print(f'PI features saved to {save_path}')
    return

if __name__ == '__main__':
    topo_features_PI('../CRC5000', 'CRC5000_PI.csv')
    topo_features_PI('../BUS250', 'BUS250_PI.csv')
    topo_features_PI('../LC25000', 'LC25000_PI.csv')