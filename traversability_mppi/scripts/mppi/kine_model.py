import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class KINE_MODEL(nn.Module):
    def __init__(self, K, T, dt, Q, Qf, R, W, robot_length, robot_width):
        super(KINE_MODEL, self).__init__()
        self.dtype          = torch.float
        self.K              = K
        self.T              = T
        self.dt             = dt
        self.robot_length   = robot_length
        self.robot_width    = robot_width

        # Cost weights
        Q = torch.diag(torch.tensor(Q, dtype=self.dtype))
        Qf = torch.diag(torch.tensor(Qf, dtype=self.dtype))
        R = torch.diag(torch.tensor(R, dtype=self.dtype))
        W = torch.diag(torch.tensor(W, dtype=self.dtype))
        
        self.Q  = nn.Parameter(Q, requires_grad=False)
        self.Qf = nn.Parameter(Qf, requires_grad=False)
        self.R  = nn.Parameter(R, requires_grad=False)
        self.W  = nn.Parameter(W, requires_grad=False)
        self.eps = 1e-6

        # Footprint points
        dx = torch.tensor([robot_length/2, robot_length/2, -robot_length/2, -robot_length/2], dtype=self.dtype)
        dy = torch.tensor([robot_width/2, -robot_width/2, -robot_width/2, robot_width/2], dtype=self.dtype)
        self.dx = nn.Parameter(dx, requires_grad=False)
        self.dy = nn.Parameter(dy, requires_grad=False)

    def forward(self, states, action):
        '''
        Estimate future states
        - action is a [B,K,T,2] tensor
        '''
        B,K,T,_ = action.shape
        pred_states = torch.zeros(B, K, T, 3, dtype=action.dtype, device=action.device)
        pred_states[:,:,0] = states
        for t in range(self.T-1):
            st = pred_states[:,:,t]
            u = action[:,:,t]

            # # Calculate traversability
            # idxu = 2 * (st[...,0] - self.grid_bounds['xbound'][0]) / (self.grid_bounds['xbound'][1] - self.grid_bounds['xbound'][0]) - 1
            # idxv = 2 * (st[...,1] - self.grid_bounds['ybound'][0]) / (self.grid_bounds['ybound'][1] - self.grid_bounds['ybound'][0]) - 1

            # grid = torch.stack((-idxv, -idxu), -1).view(B,K,1,2)
            # traversability = F.grid_sample(travmap, grid, align_corners=True)

            # mu = traversability[:,0].view(B,K)
            # nu = traversability[:,1].view(B,K)
            
            # k1 = self.x_dot(st, u, mu, nu)
            # st2 = st + self.dt/2 * k1
            # k2 = self.x_dot(st2, u, mu, nu)
            # st3 = st + self.dt/2 * k2
            # k3 = self.x_dot(st3, u, mu, nu)
            # st4 = st + self.dt * k3
            # k4 = self.x_dot(st4, u, mu, nu)

            mu = torch.Tensor([1.0]).to(action.device)
            nu = torch.Tensor([1.0]).to(action.device)

            # pred_states[:,:,t+1] = st + (self.dt / 6) * (k1 + 2*k2 + 2*k3 + k4)
            pred_states[:,:,t+1] = st + self.x_dot(st, u, mu, nu) * self.dt

        return pred_states

    def cost_function(self, pred_states, action, reference, trav_map, grid_bounds):
        '''
        pred_states is a [B,K,T,3] tensor
        action is a [B,K,T,2] tensor
        rerence is a [B,1,T,5] tensor
        trav_map is a [B,H,W,2] tensor
        '''
        B,K,T,_ = pred_states.shape

        # Calculate states-reference cost
        states_error = (pred_states - reference[...,:3])
        dist = (states_error[...,:1,0,None]**2 + states_error[...,:1,1,None]**2)**0.5
        states_error[...,:2] = states_error[...,:2]/(dist + self.eps)

        # We need to correct for heading error [-pi,pi)
        states_error[...,2] = (states_error[...,2] + np.pi) % (2*np.pi) - np.pi
        states_cost = torch.square(states_error[...,:-1,:]) @ self.Q
        states_cost = torch.sum(states_cost, dim=-1)

        # Final state-reference cost
        states_cost_f = torch.square(states_error[...,-1:,:]) @ self.Qf
        states_cost_f = torch.sum(states_cost_f, dim=-1)
        states_cost = torch.concat([states_cost, states_cost_f], dim=-1)

        # Calculate action cost
        action_error = (action - reference[...,3:])
        action_cost = torch.square(action_error) @ self.R
        action_cost = torch.sum(action_cost, dim=-1)

        # Generate footprint points based on the predicted states
        cos_h = torch.cos(pred_states[...,2])
        sin_h = torch.sin(pred_states[...,2])

        # Rotate these points according to the robot's heading
        local_x_offsets = cos_h[..., None] * self.dx - sin_h[..., None] * self.dy
        local_y_offsets = sin_h[..., None] * self.dx + cos_h[..., None] * self.dy

        # Get footprint points in the global frame
        footprint_x = pred_states[..., 0:1] + local_x_offsets
        footprint_y = pred_states[..., 1:2] + local_y_offsets

        # Convert footprint points to normalized grid coordinates
        u = 2 * (footprint_x - grid_bounds['xbound'][0]) / (grid_bounds['xbound'][1] - grid_bounds['xbound'][0]) - 1
        v = 2 * (footprint_y - grid_bounds['ybound'][0]) / (grid_bounds['ybound'][1] - grid_bounds['ybound'][0]) - 1

        grid = torch.stack((-u, -v), -1).view(B, K*T, 4, 2)  # 4 points per footprint
        travs = F.grid_sample(trav_map, grid, align_corners=True)
        travs = travs.permute((0, 2, 3, 1))

        # Compute the map cost using the worst-case traversability
        travs, _ = torch.min(travs, dim=-2)
        trav_cost = -travs @ self.W 
        trav_cost = torch.sum(trav_cost, dim=-1).view(B, K, T)

        print('states_cost max:', states_cost.max())
        print('states_cost_f max:', states_cost_f.max())
        print('action_cost max:', action_cost.max())
        print('trav_cost min:', trav_cost.min())

        # Sum the costs
        return states_cost + action_cost + trav_cost, travs
    
    @torch.jit.script
    def x_dot(state: torch.Tensor, action: torch.Tensor, mu: torch.Tensor, nu: torch.Tensor):
        '''
        Lets define the integrator function
        '''
        # state variables
        theta = state[...,2].clone()
        lin_vel = action[...,0].clone()
        ang_vel = action[...,1].clone()

        x_dot = mu*torch.cos(theta)*lin_vel
        y_dot = mu*torch.sin(theta)*lin_vel
        theta_dot = nu*ang_vel

        st_dot = torch.stack((x_dot, y_dot, theta_dot), dim=-1)

        return st_dot