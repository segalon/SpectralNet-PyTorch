import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def orthonorm(Q, eps=1e-2):
  m = torch.tensor(Q.shape[0]) # batch size
  outer_prod = torch.mm(Q.T, Q)
  outer_prod = outer_prod + eps * torch.eye(outer_prod.shape[0])

  L = torch.linalg.cholesky(outer_prod) # lower triangular
  L_inv = torch.linalg.inv(L)
  return torch.sqrt(m) * L_inv.T


class NetOrtho(nn.Module):
    def __init__(self, params):
      super(NetOrtho, self).__init__()

      input_sz = params['input_sz']
      n_hidden_1 = params['n_hidden_1']
      n_hidden_2 = params['n_hidden_2']
      k = params['k']

      self.fc1 = nn.Linear(input_sz, n_hidden_1)
      self.fc2 = nn.Linear(n_hidden_1, n_hidden_1)
      self.fc3 = nn.Linear(n_hidden_1, n_hidden_2)
      self.fc4 = nn.Linear(n_hidden_2, k)

      self.A = torch.rand(k,k)
      self.A.requires_grad = False


    def forward(self, x, ortho_step=False):
      self.A.requires_grad = False
      if ortho_step:
        with torch.no_grad():
          x = F.relu(self.fc1(x))
          x = F.relu(self.fc2(x))
          x = F.relu(self.fc3(x))
          Y_tilde = torch.tanh(self.fc4(x))

          self.A = orthonorm(Y_tilde)
          self.A.requires_grad = False

          # for debugging
          Y = torch.mm(Y_tilde, self.A)
          res = (1/Y.shape[0]) * torch.mm(Y.T, Y)
          return res

      else:
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        Y_tilde = torch.tanh(self.fc4(x))
        # need to multiply from the right, not from the left
        Y = torch.mm(Y_tilde, self.A)
        return Y

