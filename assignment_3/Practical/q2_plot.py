import torch
import torch.autograd as autograd
import numpy as np
import q2_sampler
import matplotlib.pyplot as plt
import q2_solution
from q2_model import Critic 


class trainer():
    def __init__(self, distance, lambda_reg):
        self.distance = distance
        self.lambda_reg = lambda_reg
        self.net = Critic(2)


    def train(self, d0, d1, iterations):
        optim = torch.optim.SGD(self.net.parameters(), lr=1e-3)
        losses = []
        for it in range(iterations):
            samples_0, samples_1 = next(d0), next(d1)
            samples_0, samples_1 = torch.from_numpy(samples_0).float(), torch.from_numpy(samples_1).float()
            loss = self.get_loss(samples_0, samples_1)
            losses.append(loss)
            optim.zero_grad()
            loss.backward()
            optim.step()
        return losses
    

    def get_loss(self, samples_0, samples_1):
        loss = self.get_distance(samples_0, samples_1) + self.lambda_reg * q2_solution.lp_reg(samples_0, samples_1, self.net)
        return loss

    
    def eval_q2(self, iterations, thetas):
        d0 = q2_sampler.distribution1(0, 512)
        distances = []
        for theta in thetas:
            print(self.distance, 'Training theta: ', theta)
            d1 = q2_sampler.distribution1(theta, 512)
            self.train(d0, d1, iterations)
            samples_0, samples_1 = next(d0), next(d1)
            samples_0, samples_1 = torch.from_numpy(samples_0).float(), torch.from_numpy(samples_1).float()
            distance = self.get_distance(samples_1, samples_0)
            distances.append(distance)
        return distances


    def get_distance(self, samples_0, samples_1):
        if self.distance == "sh":
            dist = -1 * q2_solution.vf_squared_hellinger(samples_0, samples_1, self.net)
        else:
            dist = q2_solution.vf_wasserstein_distance(samples_0, samples_1, self.net)
        return dist


if __name__ == '__main__':
    lambda_reg_lp = 50
    iterations = 4000
    theta_values = np.arange(0, 2, 0.1)

    runnerEM = trainer(distance='em', lambda_reg = lambda_reg_lp)
    distancesEM = runnerEM.eval_q2(iterations, theta_values)

    plt.plot(theta_values, distancesEM, '.')
    plt.title('Estimated distance of the Wasserstein distance w.r.t theta')
    plt.ylabel('Distance')
    plt.xlabel('Theta')
    plt.show()
    plt.savefig('q2_5w.png')
    plt.close()
    plt.clf()

    runnerSH = trainer(distance='sh', lambda_reg=lambda_reg_lp)
    distancesSM = runnerSH.eval_q2(iterations, theta_values)
    plt.plot(theta_values, distancesSM, '.')
    plt.ylabel('Distance')
    plt.xlabel('Theta')
    plt.title('Estimated distance of Squared Hellinger w.r.t theta')
    plt.show()
    plt.savefig('q2_sh.png')
    plt.close()
    plt.clf()
    


