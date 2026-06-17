### ------ Get the project root folder (parent of the Code folder)----------- ###
import os
import sys
import pandas as pd
from mypackage.paths import DATA_DIR, IMAGES_DIR

import streamlit as st

# print("DATA_DIR123:", DATA_DIR)
main_path = os.path.join(DATA_DIR)


### ------------------------- Load datasets ------------------------------ ###
class LoadData:
    def __init__(self, dataset_name=None):
        self.dataset_name = dataset_name
        self.data = self.load_dataset()

    def load_dataset(self):
        # if self.dataset_name is None:
        #     return None   
        # if "SHHS" in self.dataset_name:
        #     name_dataset = "SHHS"
        # elif "WSC" in self.dataset_name:
        #     name_dataset = "WSC"

        # file_path_x = os.path.join(main_path, f"df_x_external_{name_dataset}.csv")
        # file_path_y = os.path.join(main_path, f"df_y_external_{name_dataset}.csv")
        file_path_x = os.path.join(main_path, f"df_X.csv")
        file_path_y = os.path.join(main_path, f"df_y.csv")

        df_x = pd.read_csv(file_path_x)
        df_y = pd.read_csv(file_path_y)

        return df_x, df_y
    
    def load_image_path(self, image_name):
        image_path = os.path.join(IMAGES_DIR, image_name)
        return image_path