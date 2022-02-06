import os
import argparse
import tarfile

import pandas as pd
from sklearn.model_selection import train_test_split

from torchtext.utils import download_from_url



def download_and_split_data(cfg) -> None:
    """
    Download tweet image data(tar) and csv file. Thanks to @sayakpaul who organized download setting already.
    
    :param cfg: data config that contains directory of data and random state number
    
    """
    if not os.path.exists(cfg.dir_data):
        os.mkdir(cfg.dir_data)

    image_url = "https://github.com/sayakpaul/Multimodal-Entailment-Baseline/releases/download/v1.0.0/tweet_images.tar.gz"

    download_from_url(image_url, path=None, root=cfg.dir_data, overwrite=False, hash_value=None, hash_type='sha256')

    tar = tarfile.open(f"{cfg.dir_data}/tweet_images.tar.gz", "r:gz")
    tar.extractall(cfg.dir_data)
    tar.close()
    

    df = pd.read_csv("https://github.com/sayakpaul/Multimodal-Entailment-Baseline/raw/main/csvs/tweets.csv")

    image_base_path = f"{cfg.dir_data}/tweet_images/"
    images_one_paths = []
    images_two_paths = []


    for idx in range(len(df)):
        current_row = df.iloc[idx]
        id_1 = current_row["id_1"]
        id_2 = current_row["id_2"]
        extentsion_one = current_row["image_1"].split(".")[-1] #jpg
        extentsion_two = current_row["image_2"].split(".")[-1]

        image_one_path = os.path.join(image_base_path, str(id_1) + f".{extentsion_one}") #ex) 'data/tweet_images/1375936088968200205.jpg'
        image_two_path = os.path.join(image_base_path, str(id_2) + f".{extentsion_two}")

        images_one_paths.append(image_one_path)
        images_two_paths.append(image_two_path)

    df["image_1_path"] = images_one_paths
    df["image_2_path"] = images_two_paths

    # Define label map
    label_map = {"Contradictory": 0, "Implies": 1, "NoEntailment": 2}
    
    # Create another column containing the integer ids of the string labels.
    df["label_idx"] = df["label"].apply(lambda x: label_map[x])

    # Stratified split (consider number of labels in train/val/test set)
    ## 10% for test
    train_df, test_df = train_test_split(df, test_size=0.1, stratify=df["label"].values, random_state=cfg.random_state_num)
    ## 4.5% for validation, 85.5% for train
    train_df, val_df = train_test_split(train_df, test_size=0.05, stratify=train_df["label"].values, random_state=cfg.random_state_num)

    # Save train/valid/test df as pickle files
    train_df.to_pickle(f"{cfg.dir_data}/train.pkl")
    val_df.to_pickle(f"{cfg.dir_data}/valid.pkl")
    test_df.to_pickle(f"{cfg.dir_data}/test.pkl")



if __name__ == '__main__':
    print(f">>>>>>> Welcome to Multimodal Entailment! Let's split the dataset first:)")
    print(f">>>>>>> You do not need to re-execute this code after one execution.")
        
    parser = argparse.ArgumentParser(description='Configs for splitting data')

    parser.add_argument("--dir_data", type=str, default="./data")
    parser.add_argument("--random_state_num", type=int, default=42)

    data_config = parser.parse_args()

    download_and_split_data(data_config)

    print(f">>>>>>> Splitting data is done~! Please check whether there are 3 data files(train, valid, test) in the 'data' directory.")


