This study proposes ToBaFu, a technical framework for 2D cancer image classification, comprising two innovative modules: the topology-based Topo model and the Modified ResNet model. 

We train and test the proposed models using three different cancer image datasets: the LC-25000 lung and colon cancer histopathological image dataset, the CRC-5000 colorectal cancer histological image dataset, and the BUS-250 breast ultrasound image dataset. Experimental results show ToBaFu's excellent performance across the three datasets.

1.The LC-25000 dataset. The dataset is publicly available on https://www.kaggle.com/datasets/andrewmvd/lung-and-colon-cancer-histopathological-images/data. It consists of 25,000 RGB histopathology images of lung and colon tissues, split equally into five classes. The images are 768 × 768 pixels in size and are in *.jpeg format. The five classes are: colon adenocarcinomas, benign colonic tissues, lung adenocarcinomas, lung squamous cell carcinomas and benign lung tissues.

Reference: Borkowski AA, Bui MM, Thomas LB, Wilson CP, DeLand LA, Mastorides SM. Lung and Colon Cancer Histopathological Image Dataset (LC25000). arXiv:1912.12142v1 [eess.IV], 2019.

2.The CRC-5000 dataset. The public dataset is available at https://zenodo.org/records/53169. It contains 5000 colorectal cancer histological images and 8 different tissue classes (TUMOR, STROMA, COMPLEX, LYMPHO, DEBRIS, MUCOSA, ADIPOSE, EMPTY), every class comprises 625 images and each image is saved in *.tif format.

Reference: Kather JN, Weis CA, Bianconi F, Melchers SM, Schad LR, Gaiser T, Marx A, Zollner F: Multi-class texture analysis in colorectal cancer histology (2016), Scientific Reports.

3.The BUS-250 dataset. The public dataset can be accessed at https://data.mendeley.com/datasets/wmy84gzngw/1. It comprises 250 grayscale breast ultrasound images, including 100 benign samples and 150 malignant samples. Images are stored in *.bmp format. Given different pixel sizes of the images—ranging from a minimum of 93 × 57 to a maximum of 199 × 161, we resize each original image size to 100 × 75.

Reference: Rodrigues, Paulo Sergio (2018), "Breast Ultrasound Image", Mendeley Data, V1, doi: 10.17632/wmy84gzngw.1