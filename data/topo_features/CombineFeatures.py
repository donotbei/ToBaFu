# Combine Statistic features and PI features

import pandas as pd


def merge_files(pi_path, topo_features_path, output_path):
    pi_test_df = pd.read_csv(pi_path)
    topo_features_test_df = pd.read_csv(topo_features_path)
    topo_features_test_df = topo_features_test_df.iloc[:, :-61]
    if 'image_path' in topo_features_test_df.columns:
        topo_features_test_df = topo_features_test_df.drop(columns=['image_path'])
    topo_features_test_df.insert(0, 'image_path', pi_test_df['image_path'])
    combined_df = pd.concat([topo_features_test_df, pi_test_df.iloc[:, 1:]], axis=1)
    combined_df.to_csv(output_path, index=False)
    print("A new CSV file has been generated:", output_path)


if __name__ == '__main__':
    # Combine Statistic and PI features
    merge_files('BUS250_PI.csv', 'BUS250_S.csv', 'BUS250_F.csv')
    merge_files('CRC5000_PI.csv', 'CRC5000_S.csv', 'CRC5000_F.csv')
