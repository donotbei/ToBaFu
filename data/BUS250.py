import pandas as pd


data_path = 'topo_features/CRC5000_S.csv'
# 读取BUS_250.csv文件
data = pd.read_csv(data_path)
# # 设置字典，将标签映射成类别名
# label_map = {1: '01_TUMOR', 2: '02_STROMA', 3: '03_COMPLEX', 4: '04_LYMPHO',
#              5: '05_DEBRIS', 6: '06_MUCOSA', 7: '07_ADIPOSE', 8: '08_EMPTY'}
# # 根据标签列修改image_path, 使其符合文件路径
# data['image_path'] = data['label'].map(label_map) + '/' + data['image_path']
# # 保存到新的CSV文件
# data.to_csv(data_path, index=False)
print(data.head())
