from torchvision import transforms, datasets
from torchvision.transforms.transforms import CenterCrop, datasets

root_dir = './data/'

train_transforms = transforms.Compose( [transforms.resize(224), 
                                        transforms.RandomRotation(30),
                                        transforms.RandomResizedCrop(224),
                                        transforms.RandomHorizontalFlip(),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]) ])

test_transforms = transforms.Compose( [transforms.resize(225),
                                        transforms.CenterCrop(224),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.485, 0.456, 0.406],
                                                             [0.229, 0.224, 0.225]) ])

train_dataset = datasets.ImageFolder(root=root_dir + 'train/', transform=train_transforms)
test_dataset = datasets.ImageFolder(root=root_dir + 'test/', transform=test_transforms)