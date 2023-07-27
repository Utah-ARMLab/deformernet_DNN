class PointNetShapeServoMDN(nn.Module):
    def __init__(self, normal_channel=True, n_gaussians=2, output_dim=3):
        super(PointNetShapeServoMDN, self).__init__()
        
        if normal_channel:
            additional_channel = 3
        else:
            additional_channel = 0
        self.normal_channel = normal_channel
        self.sa1 = PointConvDensitySetAbstraction(npoint=512, nsample=32, in_channel=6+additional_channel, mlp=[64], bandwidth = 0.1, group_all=False)
        self.sa2 = PointConvDensitySetAbstraction(npoint=128, nsample=64, in_channel=64 + 3, mlp=[128], bandwidth = 0.2, group_all=False)
        self.sa3 = PointConvDensitySetAbstraction(npoint=1, nsample=None, in_channel=128 + 3, mlp=[256], bandwidth = 0.4, group_all=True)

        self.fc1 = nn.Linear(256, 128)
        self.bn1 = nn.GroupNorm(1, 128)



        self.fc3 = nn.Linear(128, 64)
        self.bn3 = nn.GroupNorm(1, 64)
        self.fc4 = nn.Linear(64, 32)
        self.bn4 = nn.GroupNorm(1, 32)
        self.fc5 = nn.Linear(32, 3)


        # MDN parameters
        self.pi = nn.Linear(last_layer_size, self.n_gaussians)
        self.mu = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs) 

        # lower triangle of covariance matrix (below the diagonal)
        self.L = nn.Linear(last_layer_size, int(0.5 * self.n_gaussians * self.n_outputs * (self.n_outputs - 1)))
        # the diagonal of covariance matrix
        self.L_diagonal = nn.Linear(last_layer_size, self.n_gaussians * self.n_outputs)






    def forward(self, xyz, xyz_goal):
        # Set Abstraction layers
        B,C,N = xyz.shape
        if self.normal_channel:
            l0_points = xyz
            l0_xyz = xyz[:,:3,:]
        else:
            l0_points = xyz
            l0_xyz = xyz
        
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        
        x = l3_points.view(B, 256)

        if self.normal_channel:
            l0_points = xyz_goal
            l0_xyz = xyz_goal[:,:3,:]
        else:
            l0_points = xyz_goal
            l0_xyz = xyz_goal
        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)        
        g = l3_points.view(B, 256)

        x = g - x
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn3(self.fc3(x)))
        x = F.relu(self.bn4(self.fc4(x)))
        # x = self.fc5(x)

        pi = nn.functional.softmax(self.pi(x), -1)
        mu = self.mu(x).reshape(-1, self.n_outputs, self.n_gaussians)
        L_diagonal = torch.exp(self.L_diagonal(x)).reshape(-1, self.n_outputs, self.n_gaussians)

        # below the main diagonal
        L = self.L(x).reshape(-1, int(0.5 * self.n_outputs * (self.n_outputs - 1)), self.n_gaussians)

        return pi, mu, L, L_diagonal

    def loss_fn(self, pi, mu, L, L_d, target, save_input_gradient = False):
        if save_input_gradient:
            target.requires_grad = True
        result = torch.zeros(target.shape[0], self.n_gaussians).to(self.device)
        tril_idx = np.tril_indices(self.n_outputs, -1) # -1 because it's below the main diagonal
        diag_idx = np.diag_indices(self.n_outputs)

        for idx in range(self.n_gaussians):
            tmp_mat = torch.zeros(target.shape[0], self.n_outputs, self.n_outputs).to(self.device)
            tmp_mat[:, tril_idx[0], tril_idx[1]] = L[:, :, idx]
            tmp_mat[:, diag_idx[0], diag_idx[1]] = L_d[:, :, idx]
            mvgaussian = MultivariateNormal(loc=mu[:, :, idx], scale_tril=tmp_mat)
            result_per_gaussian = mvgaussian.log_prob(target)
            result[:, idx] = result_per_gaussian + pi[:, idx].log()
        result = -torch.mean(torch.logsumexp(result, dim=1))

        # when optimizing over q using non-torch optimizer (check planner.py)
        if save_input_gradient:
            result.backward(retain_graph=True)
            self.q_grad = target.grad.cpu().numpy().astype('float64')

        return result