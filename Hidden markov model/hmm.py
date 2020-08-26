from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array alpha[i, t] = P(Z_t = s_i, x_1:x_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        Osequence_index = np.array([self.obs_dict[item] for item in Osequence])
        ###################################################
        # Edit here
        for i in range(S):
            alpha[i,0]= self.pi[i]*self.B[i,Osequence_index[0]]
        for j in range(1,L):
            for i in range(S):
                alpha[i,j] = self.B[i,Osequence_index[j]]* sum(self.A[k,i]*alpha[k,j-1] for k in range(S))
        ###################################################
        return alpha

    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(Z_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(Z_t = s_j|Z_t-1 = s_i)
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, k] = P(X_t = o_k| Z_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array beta[i, t] = P(x_t+1:x_T | Z_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        Osequence_index = np.array([self.obs_dict[item] for item in Osequence])
        beta[:, L-1] = 1
        j = L-2
        while j>=0:
            for i in range(S):
                beta[i,j] = sum(self.A[i,k]*self.B[k,Osequence_index[j+1]]*beta[k,j+1] for k in range(S))
            j=j-1
        ###################################################
        return beta

    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(x_1:x_T | λ)
        """
        prob = 0

        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        prob = sum(alpha[:,-1])
        ###################################################
        return prob

    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(s_t = i|O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        gama = self.sequence_prob(Osequence)
        for j in range (0,L):
            for i in range(0,S):
                prob[i,j] = alpha[i,j]*beta[i,j]/gama
        ###################################################
        return prob
    #TODO:
    def likelihood_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*num_state*(L-1)) A numpy array of P(X_t = i, X_t+1 = j | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, S, L - 1])
        ###################################################
        # Edit here
        Osequence_index = np.array([self.obs_dict[item] for item in Osequence])
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        gama = self.sequence_prob(Osequence)
        for i in range(0,S):
            for j in range(0,S):
                for k in range(0,L-1):
                    prob[i,j,k] = alpha[i,k]* self.A[i,j]* self.B[j,Osequence_index[k+1]]*beta[j,k+1]/gama

        ###################################################
        return prob

    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []

        ###################################################
        # Q3.3 Edit here
        S = len(self.pi)
        L = len(Osequence)
        sigma = np.zeros([S, L])
        delta = np.zeros([S, L],dtype="int")
        path_temp = np.zeros([L],dtype="int")
        Osequence_index = np.array([self.obs_dict[item] for item in Osequence])
        for i in range(0,S):
            sigma[i,0]= self.pi[i]* self.B[i,Osequence_index[0]]
        for j in range(1,L):
            for i in range(0,S):
                sigma[i,j]= self.B[i,Osequence_index[j]]* max(self.A[k,i]* sigma[k,j-1] for k in range(0,S))
                delta[i,j]= np.argmax([self.A[k,i]* sigma[k,j-1] for k in range(0,S)])

        path_temp[L-1]= np.argmax(sigma[:,L-1])
        print('delta',delta)
        for m in reversed(range(1,L)):
            path_temp[m-1] = delta[path_temp[m],m]
        path = path_temp.tolist()
        j=0

        for i in range(0,L):
            for key, value in self.state_dict.items():
                if value == path[i]:
                    path[j]=key
            j=j+1


        ###################################################
        return path
