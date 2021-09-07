import pandas as pd
from torch.utils.data import Dataset, DataLoader, dataloader

data_path = "https://people.sc.fsu.edu/~jburkardt/data/csv/homes.csv"

class CustomDataset(Dataset):
    
    def __init__(self, data_path, budget = 152):
        
        df = pd.read_csv(data_path)
        df.columns = [x.replace('"', '').replace(' ', '') for x in df.columns]
        columns = ["Living", "Rooms", "Beds", "Baths", "Age", "Acres", "Taxes"]

        self.X = df[columns].values
        self.y = (df["Sell"] <= budget).astype("int")
        self.n_samples = self.X.shape[0]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, index):
        return self.X[index], self.y[index]

data = CustomDataset(data_path, budget = 160)

#Shuffle true is always important so the algorithm doesn't learn the position of the labels
data1 = DataLoader(dataset = data, batch_size = 10, shuffle = True)

dataiter = iter(data1).next()

print(dataiter)
print(len(dataiter[0]))