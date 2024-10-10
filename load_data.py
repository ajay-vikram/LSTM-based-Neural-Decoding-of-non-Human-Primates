import torch
from neurobench.datasets import PrimateReaching
from torch.utils.data import DataLoader, Subset
from torch.utils.data import random_split
from torch.utils.data import TensorDataset, DataLoader


all_files = ["indy_20160622_01", "indy_20160630_01", "indy_20170131_02", 
             "loco_20170210_03", "loco_20170215_02", "loco_20170301_05"]

dataset = []

def normalize_dataset(dataset):
    normalized_dataset = []
    for sample, label in dataset:
        # if torch.all(sample == 0):
        #     normalized_sample = torch.zeros_like(sample)
        # else:
        #     normalized_sample = (sample - torch.mean(sample)) / torch.std(sample)
        normalized_dataset.append((sample, label))
    return normalized_dataset

if __name__ == "__main__":
    for filename in all_files:
        print("Processing {}".format(filename))

        # The dataloader and preprocessor has been combined together into a single class
        dataset = PrimateReaching(file_path=data_dir, filename=filename,
                                num_steps=1, train_ratio=0.5, bin_width=0.084,
                                biological_delay=0, remove_segments_inactive=False, download=True)
        normalized_dataset = normalize_dataset(dataset)

        train_dataloader = DataLoader(Subset(normalized_dataset, dataset.ind_train), batch_size=len(dataset.ind_train), shuffle=False)
        val_dataloader = DataLoader(Subset(normalized_dataset, dataset.ind_val), batch_size=len(dataset.ind_val), shuffle=False)
        test_dataloader = DataLoader(Subset(normalized_dataset, dataset.ind_test), batch_size=len(dataset.ind_test), shuffle=False)

        torch.save(train_dataloader, store_dir + filename + "/train.pth")
        torch.save(val_dataloader, store_dir + filename + "/val.pth")
        torch.save(test_dataloader, store_dir + filename + "/test.pth")