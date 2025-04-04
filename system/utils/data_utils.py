import numpy as np
import os
import torch

from PIL import Image
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset, Dataset


def read_data(dataset, idx, is_train=True):
    if is_train:
        train_data_dir = os.path.join('../dataset', dataset, 'train/')

        train_file = train_data_dir + str(idx) + '.npz'
        with open(train_file, 'rb') as f:
            train_data = np.load(f, allow_pickle=True)['data'].tolist()

        return train_data

    else:
        test_data_dir = os.path.join('../dataset', dataset, 'test/')

        test_file = test_data_dir + str(idx) + '.npz'
        with open(test_file, 'rb') as f:
            test_data = np.load(f, allow_pickle=True)['data'].tolist()

        return test_data


def read_client_data(dataset, idx, is_train=True):
    if dataset[:2] == "ag" or dataset[:2] == "SS":
        return read_client_data_text(dataset, idx, is_train)
    elif dataset[:2] == "sh":
        return read_client_data_shakespeare(dataset, idx)

    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.float32)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.float32)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data


def read_client_data_text(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train, X_train_lens = list(zip(*train_data['x']))
        y_train = train_data['y']

        X_train = torch.Tensor(X_train).type(torch.int64)
        X_train_lens = torch.Tensor(X_train_lens).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [((x, lens), y) for x, lens, y in zip(X_train, X_train_lens, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test, X_test_lens = list(zip(*test_data['x']))
        y_test = test_data['y']

        X_test = torch.Tensor(X_test).type(torch.int64)
        X_test_lens = torch.Tensor(X_test_lens).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)

        test_data = [((x, lens), y) for x, lens, y in zip(X_test, X_test_lens, y_test)]
        return test_data


def read_client_data_shakespeare(dataset, idx, is_train=True):
    if is_train:
        train_data = read_data(dataset, idx, is_train)
        X_train = torch.Tensor(train_data['x']).type(torch.int64)
        y_train = torch.Tensor(train_data['y']).type(torch.int64)

        train_data = [(x, y) for x, y in zip(X_train, y_train)]
        return train_data
    else:
        test_data = read_data(dataset, idx, is_train)
        X_test = torch.Tensor(test_data['x']).type(torch.int64)
        y_test = torch.Tensor(test_data['y']).type(torch.int64)
        test_data = [(x, y) for x, y in zip(X_test, y_test)]
        return test_data

def load_dataset_representations(folder_name, batch_size=32, n_classes = 10, shuffle=True):
    transform = transforms.ToTensor()
    folder_name = os.path.join('../dataset', folder_name, 'representations/')

    images = []
    labels = []
    for label in range(n_classes):  # MNIST tem 10 classes (0-9)
        label_folder = os.path.join(folder_name, str(label))
        if not os.path.exists(label_folder):
            continue

        for file_name in os.listdir(label_folder):
            file_path = os.path.join(label_folder, file_name)
            image = Image.open(file_path).convert('L')  # Converte para escala de cinza
            image = transform(image)  # Converte a imagem PIL para tensor
            images.append(image)
            labels.append(label)
    
    if len(images) == 0:
        raise RuntimeError(f"Nenhuma imagem encontrada no diretório {folder_name}")
    
    images = torch.stack(images)  # Converte lista de tensores para um tensor
    labels = torch.tensor(labels)
    
    dataset = TensorDataset(images, labels)
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    
    return data_loader

class CustomDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        label = self.labels[idx]
        return image, label

def load_validate_server(size_batch, dataset):
    # Passo 1: Carregar os dados do arquivo .npz
    load_path = os.path.join('../dataset', dataset, 'validate', 'validation_data.npz')

    data = np.load(load_path)

    data_validate = data['data']
    label_validate = data['labels']

    validation_dataset = CustomDataset(data_validate, label_validate)
    validation_loader = DataLoader(validation_dataset, batch_size=size_batch, shuffle=True)

    return validation_loader
