import torch
import sys
import numpy as np

class CudaCKA(object):
    def __init__(self, device):
        self.device = device

    def centering(self, K):
        n = K.shape[0]
        unit = torch.ones([n, n], device=self.device)
        I = torch.eye(n, device=self.device)
        H = I - unit / n
        return torch.matmul(torch.matmul(H, K), H)

    def linear_HSIC(self, X, Y):     
        L_X = torch.matmul(X, X.T)
        L_Y = torch.matmul(Y, Y.T)
        return torch.sum(self.centering(L_X) * self.centering(L_Y))

    def linear_CKA(self, X, Y):
        hsic = self.linear_HSIC(X, Y)
        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))

        if hsic == torch.inf:
            return 1
        elif (hsic / (var1 * var2)) == torch.nan:
            return 1
        
        return hsic / (var1 * var2)
    
    def calcule_cka(self, model1, model2, dataloader, algo):

        X = get_representations(model=model1, data_loader=dataloader,device=self.device)
        Y = get_representations(model=model2, data_loader=dataloader,device=self.device)
        
        if algo == "FedAvg":
            X = torch.tensor(X[0])
            Y = torch.tensor(Y[0])
        elif algo == "FedALA":
            X = X[0][0]
            Y = Y[0][0]
        
        linear_cka = self.linear_CKA(X.view(-1, 1), Y.view(-1, 1))
        if not np.isnan(linear_cka.to('cpu')):
            return self.linear_CKA(X.view(-1, 1), Y.view(-1, 1))
        else:
            return 0


def get_representations(model, data_loader, device):
    model.eval()
    representations = []  # Ajuste conforme o número de camadas que deseja capturar


    with torch.no_grad():
        data = next(iter(data_loader))
        size = data[0][0].unsqueeze(0).shape
        
        torch.manual_seed(0)
        inputs = torch.rand(size)
        inputs = inputs.to(device)

        out = model.forward_representations(inputs)  # Executa a passagem dos dados pelo modelo
        representations.append(out)

    return representations



if __name__ == "__main__":
    pass
