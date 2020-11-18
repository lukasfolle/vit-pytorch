import os
import pandas as pd
from random import shuffle

from pytorch_lightning import LightningDataModule
from monai.data import CacheDataset
from monai.transforms import Compose

from experiments.Transforms import PNGsToStack, Normalize
from experiments.Dataloader import CovidDataLoader
from experiments.SliceViewer import scroll_slices


def path_exists(path):
    if os.path.exists(path):
        return True
    return False


class CovidCT(LightningDataModule):
    def __init__(self, base_path="/home/lukas/Documents/Datasets/covid_ctscan/New_Data_CoV2", split=[0.7, 0.2, 0.1],
                 batch_size=8, num_workers=8):
        super().__init__()
        self.base_path = base_path
        self.split = split
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.df = self.create_df()
        self.df_monai = self.get_monai_format()
        shuffle(self.df_monai, lambda: 0.42)

    def create_df(self):
        diag_classes = ["Covid", "Healthy", "Others"]
        class_int_mapping = {"Healthy": 0, "Covid": 1, "Others": 2}
        df = pd.DataFrame(columns=["path", "class"])
        for diag_class in diag_classes:
            class_path = os.path.join(self.base_path, diag_class)
            patients = os.listdir(class_path)
            for patient in patients:
                patient_path = os.path.join(class_path, patient)
                if path_exists(patient_path):
                    df = df.append({"path": patient_path, "class": class_int_mapping[diag_class]}, ignore_index=True)
                else:
                    print(f"INFO: Could not find path {patient_path}.")
        return df

    def get_monai_format(self):
        df_dict = self.df.to_dict()
        monai_format = []
        for idx in range(len(df_dict["path"])):
            item = {"volume": df_dict["path"][idx], "label": df_dict["class"][idx]}
            monai_format.append(item)
        return monai_format

    @staticmethod
    def get_cache_covid_ds(ds):
        transform = Compose([PNGsToStack(["volume"]),
                             Normalize(["volume"])])
        return CacheDataset(ds, transform, num_workers=8)

    def train_dataloader(self):
        ds = self.df_monai[:int(self.split[0] * len(self.df_monai))]
        ds = self.get_cache_covid_ds(ds)
        return CovidDataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        ds = self.df_monai[int(self.split[0] * len(self.df_monai)):
                           int(self.split[0] * len(self.df_monai)) + int(self.split[1] * len(self.df_monai))]
        ds = self.get_cache_covid_ds(ds)
        return CovidDataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        ds = self.df_monai[-int(self.split[2] * len(self.df_monai)):]
        ds = self.get_cache_covid_ds(ds)
        return CovidDataLoader(ds, batch_size=self.batch_size, num_workers=self.num_workers, pin_memory=True)


if __name__ == "__main__":
    ds = CovidCT()
    ds.test_dataloader()
