import torch

class CKA():
    def __init__(self, device):
        self.device = device

    def linear_HSIC(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)
        
        x_center = (X.t() - X.mean())
        y_center = (Y.t() - Y.mean())
        cov_xy = torch.sum(torch.mm(x_center, y_center.t()))
        return torch.abs(cov_xy) ** 2

    def linear_CKA(self, X, Y):
        X = X.to(self.device)
        Y = Y.to(self.device)

        hsic = self.linear_HSIC(X, Y)

        var1 = torch.sqrt(self.linear_HSIC(X, X))
        var2 = torch.sqrt(self.linear_HSIC(Y, Y))
        cka = hsic / (var1 * var2)
        # Evita divisão por zero
        if var1 == 0 or var2 == 0:
            return torch.tensor(1.0, device=self.device)  # ou um valor que faça sentido para o seu caso
        if cka == torch.nan:
            return torch.tensor(1.0, device=self.device)

        return hsic / (var1 * var2)