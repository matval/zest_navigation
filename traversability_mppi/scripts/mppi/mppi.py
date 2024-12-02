'''
Code adapted from https://github.com/UM-ARM-Lab/pytorch_mppi
'''
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.distributions.normal import Normal

from .kine_model import KINE_MODEL

class MPPI(nn.Module):
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.
    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dt, N, K, gamma, noise_mu, noise_sigma, u_min, u_max, Q, Qf, R, W, robot_length, robot_width):
        super(MPPI, self).__init__()
        self.dtype  = torch.float
        self.dt     = dt    # DELTA_TIME
        self.K      = K     # N_SAMPLES
        self.T      = N     # TIMESTEPS
        self.u_min  = u_min # Min control input
        self.u_max  = u_max # Max control input
        self.m      = 2     # action_dim
        self.gamma  = gamma # Optimization temperature
        
        # initial solution
        self.u_init = torch.zeros((self.T, self.m), dtype=self.dtype).cuda()

        noise_mu = torch.tensor(noise_mu, dtype=self.dtype).cuda()
        noise_sigma = torch.tensor(noise_sigma, dtype=self.dtype).cuda()
        self.noise_sigma_inv = torch.inverse(torch.diag(noise_sigma))
        self.noise_dist = Normal(loc=noise_mu, scale=noise_sigma)

        # T x nu control sequence
        self.U = self.u_init

        # Initialize dynamic model
        self.dyn_model = KINE_MODEL(
            K=K,
            T=N,
            dt=dt,
            Q=Q,
            Qf=Qf,
            R=R,
            W=W,
            robot_length=robot_length,
            robot_width=robot_width
        )

    def forward(self, states, reference, trav_map, grid_bounds):
        B,_,_ = reference.shape
        # resample noise each time we take an action; these can be done at the start
        noise = self.noise_dist.sample((B, self.K, self.T))
        
        # Generate perturbed actions        
        perturbed_action = self.U + noise
        perturbed_action[...,0] = torch.clamp(perturbed_action[...,0], min=self.u_min[0], max=self.u_max[0])
        perturbed_action[...,1] = torch.clamp(perturbed_action[...,1], min=self.u_min[1], max=self.u_max[1])

        # Get the predicted states
        pred_states = self.dyn_model(states, perturbed_action)

        # Calculate costs
        costs, travs = self.dyn_model.cost_function(pred_states, perturbed_action, reference.unsqueeze(1), trav_map, grid_bounds)
        pert_costs = self.perturbation_cost(perturbed_action)
        costs = costs + pert_costs

        # MPPI update
        scores = -torch.sum(costs, dim=-1)
        probs = torch.exp(self.gamma * (scores - torch.max(scores, dim=-1, keepdim=True)[0]))
        probs = probs / torch.sum(probs, dim=-1, keepdim=True)
        a_hat = torch.sum(perturbed_action*probs.view((B,self.K,1,1)), dim=1)
        
        # Get the states corresponding to the best action
        states_hat = self.dyn_model(states, a_hat.unsqueeze(0)).squeeze(0)

        # Update the control sequence
        a_hat_next = torch.roll(a_hat, -1, dims=1)
        a_hat_next[:,-1] = self.u_init[-1]
        self.U = a_hat_next.detach()

        return a_hat, states_hat, pred_states, probs, travs
    
    def perturbation_cost(self, action):
        # Compute the perturbation cost
        noise = action - self.U
        action_cost = 1./self.gamma * noise @ self.noise_sigma_inv  # Like original paper

        # action perturbation cost
        perturbation_cost = torch.sum(self.U * action_cost, dim=-1)

        # Sum the costs
        return perturbation_cost

    def reset(self, u=None):
        if u is not None:
            self.U = u.unsqueeze(1)
        else:
            self.U = self.u_init