from cvxpylayers.torch import CvxpyLayer
import torch
import cvxpy as cp


class ReluLayer(torch.nn.Module):
    def __init__(self, D_in, D_out):
        super(ReluLayer, self).__init__()
        self.W = torch.nn.Parameter(1e-3*torch.randn(D_out, D_in))
        self.b = torch.nn.Parameter(1e-3*torch.randn(D_out))
        z = cp.Variable(D_out)
        Wtilde = cp.Variable((D_out, D_in))
        W = cp.Parameter((D_out, D_in))
        b = cp.Parameter(D_out)
        x = cp.Parameter(D_in)
        prob = cp.Problem(cp.Minimize(cp.sum_squares(
            z-Wtilde@x-b)), [z >= 0, Wtilde == W])
        self.layer = CvxpyLayer(prob, [W, b, x], [z])

    def forward(self, x):
        # when x is batched, repeat W and b
        if x.ndim == 2:
            batch_size = x.shape[0]
            return self.layer(self.W.repeat(batch_size, 1, 1), self.b.repeat(batch_size, 1), x)[0]
        else:
            return self.layer(self.W, self.b, x)[0]


torch.manual_seed(0)
net = torch.nn.Sequential(
    torch.nn.Linear(20, 20),
    ReluLayer(20, 20),
    ReluLayer(20, 20),
    torch.nn.Linear(20, 1)
)
X = torch.randn(300, 20)
Y = torch.randn(300, 1)


opt = torch.optim.Adam(net.parameters(), lr=1e-2)
for _ in range(25):
    opt.zero_grad()
    l = torch.nn.MSELoss()(net(X), Y)
    print(l.item())
    l.backward()
    opt.step()
