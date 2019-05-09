from utils import *

# Dataset 
def getData(mode):
    if mode == 'train':
        img = pd.read_csv('train_img.csv')
        label = pd.read_csv('train_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)
    else:
        img = pd.read_csv('test_img.csv')
        label = pd.read_csv('test_label.csv')
        return np.squeeze(img.values), np.squeeze(label.values)

class RetinopathyDataset(data.Dataset):
    def __init__(self, root, mode, transform=None):
        self.root = root
        self.transform = transform
        self.img_names, self.labels = getData(mode)
        
    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, index):
        img_name = self.img_names[index] + ".jpeg"
        label = self.labels[index]
        img_path = os.path.join(self.root, img_name)
        img = Image.open(img_path)
        if self.transform!= None:
            img = self.transform(img)
        return img, label