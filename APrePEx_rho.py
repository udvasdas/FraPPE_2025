import os
import random
import numpy as np
import pandas as pd
from scipy.optimize import linprog, minimize, LinearConstraint
#import matplotlib.pyplot as plt
import itertools
#from copy import deepcopy
import cvxpy as cp
import time
from array import array

#from pynverse import inversefunc
#from joblib import Parallel, delayed

seed = 123

PRECISION = 1e-12

#Important functions
def runif_in_simplex(n):
  ''' Return uniformly random vector in the n-simplex '''

  k = np.random.exponential(scale=1.0, size=n)
  return k / sum(k)

def get_policy(mu):
    
    simplex = np.array([1]*mu.shape[0]).reshape(1, -1)
    #print(simplex)
    eye = np.eye(mu.shape[0])
    one = np.array([1])
    A = np.concatenate([-eye, simplex, -simplex], axis=0)
    b = np.concatenate([np.zeros(len(mu)), one, -one], axis=0)
    opt_policies = []
    #print(mu[0:,0])
    for l in range(mu.shape[1]):
        results = linprog(
            -mu[0:,l], A_ub=A, b_ub=b, A_eq=None, b_eq=None, method="highs-ds"
        )  # Use simplex method
        if not results["success"]:
            raise "LP Solver failed"
        # Get active constraints
        opt_policies.append(results["x"])
        #aux = {"A": A, "b": b, "slack": results["slack"]}
        #aux.list.append(aux)
    return np.array(opt_policies)


def arreqclose_in_list(myarr, list_arrays):
    """
    Test if np array is in list of np arrays
    """
    return next(
        (
            True
            for elem in list_arrays
            if elem.size == myarr.size and np.allclose(elem, myarr)
        ),
        False,
    )


def enumerate_all_policies(A, b):
    """
    Enumerate all policies in the polytope Ax <= b
    """
    # Compute all possible bases
    n_constraints = A.shape[0]
    n_arms = A.shape[1]
    bases = list(itertools.combinations(range(n_constraints), n_arms))
    policies = []
    for base in bases:
        base = np.array(base)
        B = A[base]
        # Check that the base is not degenerate
        if np.linalg.matrix_rank(B) == A.shape[1]:
            policy = np.linalg.solve(B, b[base])
            # Verify that policy is in the polytope
            if np.all(A.dot(policy) <= b + 1e-5) and not arreqclose_in_list(
                policy, policies
            ):
                policies.append(policy)
    return policies



def compute_neighbors(vertex,pareto_arms):
    """
    Compute all neighbors of vertex in the polytope Ax <= b
    :param vertex: vertex of the polytope
    :param A: matrix of constraints
    :param b: vector of constraints
    :param slack: vector of slack variables
    """
    neighbor_list = []
    #boo = [vertex[i]==0 for i in range(len(vertex))]
    for i in range(len(vertex)):
        if vertex[i]==0 :#and i not in pareto_arms:
            arr = np.array([0]*len(vertex))
            arr[i] = 1
            neighbor_list.append(arr)
    return neighbor_list


def get_alpha(rind, W):
    """
    Compute alpha_rind for row rind of W 
    :param rind: row index
    :param W: (n_constraint,D) ndarray
    :return: alpha_rind.
    """
    m = W.shape[0]+1 #number of constraints
    D = W.shape[1]
    f = -W[rind,:]
    A = []
    b = []
    c = []
    d = []
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.zeros(1))
    
    A.append(np.eye(D))
    b.append(np.zeros(D))
    c.append(np.zeros(D))
    d.append(np.ones(1))

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    soc_constraints = [
          cp.SOC(c[i].T @ x + d[i], A[i] @ x + b[i]) for i in range(m)
    ]
    prob = cp.Problem(cp.Minimize(f.T@x),
                  soc_constraints)
    prob.solve()

    """
    # Print result.
    print("The optimal value is", -prob.value)
    print("A solution x is")
    print(x.value)
    for i in range(m):
        print("SOC constraint %i dual variable solution" % i)
        print(soc_constraints[i].dual_value)
    """    
        
    return -prob.value 

def get_bigmij(vi, vj, W):
    """
    Compute M(i,j) for designs i and j 
    :param vi, vj: (D,1) ndarrays
    :param W: (n_constraint,D) ndarray
    :return: M(i,j).
    """
    D = W.shape[1]
    P = 2*np.eye(D)
    q = (-2*(vj-vi)).ravel()
    G = -W
    h = -np.array([np.max([0,np.dot(W[0,:],vj-vi)[0]]),
                np.max([0,np.dot(W[1,:],vj-vi)[0]])])

    # Define and solve the CVXPY problem.
    x = cp.Variable(D)
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(x, P) + q.T @ x),
                 [G @ x <= h])
    #A @ x == b    
    prob.solve()
    bigmij = np.sqrt(prob.value + np.dot((vj-vi).T, vj-vi)).ravel()

    # Print result.
    #print("\nThe optimal value is", prob.value)
    #print("A solution x is")
    #print(x.value)
    #print("A dual solution corresponding to the inequality constraints is")
    #print(prob.constraints[0].dual_value)
    #print("M(i,j) is", bigmij)
    return bigmij


#Vector_optimizer
def Vect_opt(mu, W):
    is_efficient = np.arange(mu.shape[0])
    n_points = mu.shape[0]
    next_point_index = 0  # Next index in the is_efficient array to search for
    while next_point_index<len(mu):
        nondominated_point_mask = np.zeros(mu.shape[0], dtype=bool)
        vj = mu[next_point_index].reshape(-1,1)
        for i in range(len(mu)):
            vi = mu[i].reshape(-1,1)
            prod = np.matmul(W,vj-vi)
            prod[prod<0] = 0
            smallmij = (prod/[get_alpha(i, W) for i in range(W.shape[0])]).min()    
            nondominated_point_mask[i] =  (smallmij == 0) and (get_bigmij(vi, vj, W) > 0)
        nondominated_point_mask[next_point_index] = True
        is_efficient = is_efficient[nondominated_point_mask]  # Remove dominated points
        mu = mu[nondominated_point_mask]
        next_point_index = np.sum(nondominated_point_mask[:next_point_index])+1
    
    #pareto_arms = np.zeros((mu.shape[0],mu.shape[0])) 
    #for i in is_efficient:
    #    pareto_arms[i][i] = 1   
    return is_efficient

'''
def gaussian_projection(w, mu, pi1, pi2, sigma, W):
    v = pi1 - pi2
    h = -np.array([np.max([0,row.T@(mu.T@v)]) for row in W])
    def obj(z):
        return (z**2).sum() -np.dot(z,mu.T@v)
    
    z0 =  np.array([0.5]*mu.shape[1])
    
    constraints = LinearConstraint(A=-W,ub=h)#lb=np.array([0]*len(h))

    res = minimize(obj,z0,constraints=constraints)
    zinf = res.x/np.sqrt((res.x**2).sum())
    normalizer = ((v**2)/(w+PRECISION)).sum()
    var = np.sum(np.array([zinf[l]*sigma[l][l] for l in range(len(zinf))]))
    beta = np.dot(zinf,np.matmul(mu.T,v))/(var*normalizer)
    lam = np.zeros_like(mu)
    for l in range(W.shape[1]):
        lam[:,l] = mu[:,l] - (beta*sigma[l][l]*v*zinf[l])/(w+PRECISION)
    #zinf = np.array([PRECISION if zinf[i]<PRECISION else zinf[i] for i in range(len(zinf))])
    
    value = (w * ((mu@zinf - lam@zinf) ** 2)).sum() / (2 * var)
    gradient = ((mu@zinf - lam@zinf) ** 2) / (2 * var)
    return lam,value,gradient
'''

def gaussian_projection(w, mu, pi1, pi2, sigma, W):
    v = pi1 - pi2
    normalizer = ((v**2)/(w+PRECISION)).sum()
    #lagrange = mu.dot(v) / normalizer
    var = np.trace(sigma)
    lam = np.zeros_like(mu)
    
    '''
    m = W.shape[0]+1
    D = W.shape[1]
    A = []
    b = []
    c = []
    d = []
    f = mu.T@v
    #print(f)
    for i in range(W.shape[0]):
        A.append(np.zeros((1, D)))
        b.append(np.zeros(1))
        c.append(W[i,:])
        d.append(np.zeros(1))
    
    A.append(np.eye(D))
    b.append(np.zeros(D))
    c.append(np.zeros(D))
    d.append(np.ones(1))
    '''
    
    D = W.shape[1]
    P = 2*np.eye(D)
    q = (-2*(mu.T@v)).ravel()
    G = -W
    h = -np.array([np.max([0,row.T@(mu.T@v)]) for row in W])
    #print(1)
    #print(h)
    
    # Define and solve the CVXPY problem.
    z = cp.Variable(D)
    
    # We use cp.SOC(t, x) to create the SOC constraint ||x||_2 <= t.
    #soc_constraints = [cp.SOC(c[i].T @ z + d[i], A[i] @ z + b[i]) for i in range(m)
    #soc_constraints.append(cp.SOC(1,z))
    #soc_constraints.append(cp.SOC(1,-z))
    #soc_constraints.append(cp.SOC(0,cp.multiply(z,mu.T@v)))
    #print(soc_constraints)
    
    prob = cp.Problem(cp.Minimize((1/2)*cp.quad_form(z, P) + q.T @ z),
                [G @ z <= h])
    prob.solve()#solver ='ECOS'#solver ='SCS'
    zinf = z.value/(np.sqrt((z.value**2).sum()))
    #print(zinf)
    #print(f.T@zinf)
    '''
    if (zinf*(mu.T@v)).sum() > 0:
        zinf = np.array([PRECISION if zinf[i]<PRECISION else zinf[i] for i in range(len(zinf))])
        zinf = zinf/np.sqrt((zinf**2).sum())
        value = ((zinf*np.matmul(mu.T,v)).sum())**2/(2*var*(((v/(w+PRECISION))**2).sum())*((zinf**2).sum()))
        return 1,value
    else:
        return 1,1e+10
    '''
    #print((zinf**2).sum())
    #print((zinf*np.matmul(mu.T,v)).sum())
    
    var = np.dot(zinf,sigma@zinf)
    beta = np.dot(zinf,np.matmul(mu.T,v))/(var*normalizer)
    for l in range(W.shape[1]):
        lam[:,l] = mu[:,l] - (beta*sigma[l][l]*v*zinf[l])/(w+PRECISION)
    #zinf = np.array([PRECISION if zinf[i]<PRECISION else zinf[i] for i in range(len(zinf))])
    
    value = (w * ((mu@zinf - lam@zinf) ** 2)).sum() / (2 * var)
    gradient = ((mu@zinf - lam@zinf) ** 2) / (2 * var)
    return lam,value,gradient 
        
    

        
    
    #print((zinf*np.matmul(mu.T,v)).sum())
    #print(value)




def bernoulli_projection(w, mu, pi1, pi2, W, sigma=1.0):
    """
    Projection onto the hypeplane lambda^T(pi1 - pi2) = 0 assuming Bernoulli distribution using scipy minimize
    """
    mu = np.clip(mu, 1e-3, 1 - 1e-3)
    bounds = [(1e-3, 1 - 1e-3) for _ in range(len(mu))]
    v = pi1 - pi2
    constraint = LinearConstraint(v.reshape(1, -1), 0, 0)

    def objective(lam):
        kl_bernoulli = mu * np.log(mu / lam) + (1 - mu) * np.log((1 - mu) / (1 - lam))
        return (w * kl_bernoulli).sum()

    x0 = gaussian_projection(w, mu, pi1, pi2, sigma, W)[0]
    x0 = np.clip(x0, 1e-3, 1 - 1e-3)
    res = minimize(objective, x0, constraints=constraint, bounds=bounds)
    lam = res.x
    value = objective(lam)
    return lam, value


def best_response(w, mu, pi, neighbors, sigma, W, dist_type="Gaussian"):
    """
    Compute best response instance w.r.t. w by projecting onto neighbors
    :param w: weight vector
    :param mu: reward vector
    :param pi: optimal policy
    :param neighbors: list of neighbors
    :param sigma: standard deviation of Gaussian distribution
    :param dist_type: distribution type to use for projection

    return:
        - value of best response
        - best response instance
    """
    if dist_type == "Gaussian":
        projections = [
            gaussian_projection(w, mu, pi, neighbor, sigma, W) for neighbor in neighbors
        ]
    elif dist_type == "Bernoulli":
        projections = [
            bernoulli_projection(w, mu, pi, neighbor, sigma) for neighbor in neighbors
        ]
    else:
        raise NotImplementedError
    values = [p[1] for p in projections]
    instances = [p[0] for p in projections]
    grads = [p[2] for p in projections]
    return values, instances, grads


        
class Explorer:
    """
    Abstract class for an explorer
    """

    def __init__(
        self,
        n_arms,
        delta,
        sigma,
        W,
        r,
        ini_phase=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=seed,
        d_tracking=False,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts). Default: 1
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        """
        self.n_arms = n_arms
        self.delta = delta
        self.ini_phase = ini_phase
        self.sigma = sigma
        self.W = W
        self.r = r
        self.restricted_exploration = restricted_exploration
        self.dist_type = dist_type
        self.seed = seed
        #self.random_state = np.random.RandomState(seed)
        self.d_tracking = d_tracking
        self.cumulative_weights = np.zeros(n_arms)
        self.D = 1
        self.alpha = 1
        self.t = 0
        #self.empirical_allocation= np.array([0]*self.n_arms)
        self.neighbors = {}
        self.allocation = [[0]*n_arms]*n_arms
        self.means = np.zeros_like(mu)
        self.n_pulls = np.zeros(n_arms)

        if dist_type == "Gaussian":
            # Set KL divergence and lower/upper bounds for binary search
            self.kl = lambda x, y: 1 / (2 * (sigma**2)) * ((x - y) ** 2)
            self.lower = -1
            self.upper = 10
        elif dist_type == "Bernoulli":
            # Set KL divergence for Bernoulli distribution and lower/upper bounds for binary search
            self.kl = lambda x, y: x * np.log(x / y) + (1 - x) * np.log(
                (1 - x) / (1 - y)
            )
            self.lower = 0 + 1e-4
            self.upper = 1 - 1e-4
            self.ini_phase = (
                10  # Take longer initial phase for Bernoulli to avoid all 0  or all 1
            )
        else:
            raise NotImplementedError

        if restricted_exploration:
            # Compute allocation constraint
            test = np.ones_like(self.means)
            _, aux = get_policy(test)
            self.allocation_A = aux["A"]
            self.allocation_b = aux["b"]
        else:
            self.allocation_A = None
            self.allocation_b = None

    def tracking(self, allocation):
        """
        Output arm based on either d-tracking or cumulative tracking
        """
        if self.d_tracking:
            return np.argmin(self.n_pulls - self.t * allocation)
        else:
            eps = 1 / (2 * np.sqrt(self.t + self.n_arms**2))
            eps_allocation = allocation + eps
            eps_allocation = eps_allocation / eps_allocation.sum()
            self.cumulative_weights += eps_allocation
            return np.argmin(self.n_pulls - self.cumulative_weights)

    def act():
        """
        Choose an arm to play
        """
        raise NotImplementedError

    def stopping_criterion(self, vertex, allocation):
        """
        Check stopping criterion. Stopping based on the generalized log-likelihood ratio test
        """

        hash_tuple = tuple(vertex.tolist())
        #print(self.neighbors)
        game_values, conf_instances ,grads= best_response(
            w=self.empirical_allocation(),
            mu=self.means,
            pi=vertex,
            neighbors=self.neighbors[hash_tuple],
            sigma=self.sigma,
            W = self.W,
            dist_type=self.dist_type,
        )
        #beta = 3*np.sum(np.array([np.log(1 + np.log(i)) for i in self.n_pulls])) 
        beta = 2*np.log((4+np.log(self.t))/self.delta)
        #print(self.t*np.array(value_list))
        #print(self.t * game_value)
        #print(beta)
        #print("----------------")
        return self.t * np.min(game_values) > beta,self.t * np.min(game_values), beta

    def empirical_allocation(self):
        """
        Compute empirical allocation
        """
        
        return self.n_pulls / self.t
    
    def get_allocation(self, vertex, H, tol):

        if isinstance(np.sqrt(self.t/self.n_arms),int) or len(H)==0:
            z = np.array([1/self.n_arms]*self.n_arms)
        else:
            simplex = np.array([1]*self.n_arms).reshape(1, -1) 
            constraint = LinearConstraint(A=simplex, lb=1, ub=1)
            bounds = [(0, 1) for _ in range(len(mu))] 
            z0 = runif_in_simplex(n=self.n_arms) 
            value_list = []
            z_list = []     
            for h in H:
                #print(h)
                def game_obj(z):
                    return -np.dot(z-self.allocation[vertex.tolist().index(1)],h)
                res = minimize(game_obj, z0, constraints=constraint, bounds=bounds, tol=tol)
                value_list.append(-res.fun)
                z_list.append(res.x)
            z = np.array(z_list[np.argmin(np.array(value_list))])
        self.allocation[vertex.tolist().index(1)] = z/(self.t+1)+(self.t*np.array(self.allocation[vertex.tolist().index(1)]))/(self.t+1)
        return self.allocation[vertex.tolist().index(1)].tolist()
        
    def update(self, arm, reward):
        """
        Update the explorer with the reward obtained from playing the arm
        :param arm: arm played
        :param reward: reward obtained
        """
        self.t += 1
        #print(self.means)
        self.n_pulls[arm] += 1
        self.means[arm,0:] = self.means[arm,0:] + (1 / self.n_pulls[arm]) * (
            reward - self.means[arm,0:]
        )

        if self.dist_type == "Bernoulli":
            self.means = np.clip(self.means, self.lower, self.upper)


        
class FWS(Explorer):
    """
    Track-n-stop style of algorithm for bandits with preference cone
    """

    def __init__(
        self,
        n_arms,
        delta,
        sigma,
        W,
        r,
        ini_phase=1,
        restricted_exploration=False,
        dist_type="Gaussian",
        seed=seed,
        d_tracking=False,
    ):
        """
        Initialize the explorer
        :param n_arms: number of arms
        :param A: matrix constraints
        :param b: vector constraints
        :param delta: confidence parameter
        :param ini_phase: initial phase (how many times to play each arm before adaptive search starts)
        :param sigma: standard deviation of Gaussian distribution
        :param restricted_exploration: whether to use restricted exploration or not
        :param dist_type: distribution type to use for projection
        :param seed: random seed
        :param d_tracking: D-tracking or C-tracking
        """
        super().__init__(
            n_arms,
            delta,
            sigma,
            W,
            r,
            ini_phase,
            restricted_exploration,
            dist_type,
            seed,
            d_tracking,
        )

    def act(self):
        """
        Choose an arm to play
        """
        
        #print(self.means)
        if self.t < self.n_arms * self.ini_phase:
            # Initial phase play each arm once
            arm = self.t % self.n_arms
            return arm, False, None, None, None, None, None

        # Compute optimal policy w.r.t. current empirical means
        start = time.time()
        pareto_arms = Vect_opt(mu=self.means,W=W_90)
        end = time.time()
        
        #print(f"Time to get pareto set {end-start}")
        #print(pareto_arms)
        
        optimal_policy_list = np.zeros((len(pareto_arms),self.n_arms))
        #optimal_policy_list = np.eye(self.n_arms,self.n_arms)
        
        for i in range(optimal_policy_list.shape[0]):
            optimal_policy_list[i,pareto_arms[i]] = 1
            #self.allocation.append([1/self.n_arms]*self.n_arms) 
        
        #optimal_policy, aux = get_policy(mu=self.means)
        #print(A)
        #print(self.A)
        #print(optimal_policy_list)
        allocation_list = []
        stop_bool = []
        games = []
        for optimal_policy in optimal_policy_list:
        # Check if policy already visited. If yes retrieve neighbors otherwise compute neighbors
            hash_tuple = tuple(optimal_policy.tolist())  # npy not hashable
            if hash_tuple in self.neighbors:
                neighbors = self.neighbors[hash_tuple]
            else:
                neighbors = compute_neighbors(optimal_policy,pareto_arms)
                self.neighbors[hash_tuple] = neighbors
            # Solve game to get allocation
            start = time.time()
            game_values,conf_instances,grads = best_response(
                w = np.array(self.allocation[optimal_policy.tolist().index(1)]),
                mu= self.means,
                pi = optimal_policy,
                neighbors=neighbors,
                dist_type=self.dist_type,
                sigma = self.sigma,
                W=self.W)
            if len(game_values)>0:
                F_omega = np.min(game_values)
            else:
                F_omega = PRECISION
            #print(grads)
            '''
            Build sub-differential set
            '''
            self.r = self.t**(-0.9)/self.n_arms 
            H = []
            for neighbor in range(len(neighbors)):
                if game_values[neighbor]< F_omega + self.r:
                    H.append(grads[neighbor])
            '''
            Update allocation
            '''
            if not H:
                allocation = self.allocation[optimal_policy.tolist().index(1)]
            else:
                allocation = self.get_allocation(vertex=optimal_policy,H=H,tol = 1e-6)
            allocation_list.append(allocation)
            
            '''
            Check stopping for each optimal_policy
            '''
            
            stop,game,beta = self.stopping_criterion(optimal_policy,allocation)
            stop_bool.append(stop)
            games.append(game)
            game_values.append(F_omega)
            
            
        # Check if forced exploration is needed else D-tracking
        not_saturated = self.n_pulls < (np.sqrt(self.t) - self.n_arms / 2)
        if not_saturated.any() and self.d_tracking:
            # Play smallest N below sqrt(t) - n_arms/2
            arm = np.argmin(self.n_pulls)

        else:
            # Play arm according to tracking rule
            #random.seed(seed)
            allocation = random.choice(allocation_list)
            arm = self.tracking(allocation)
            
                
        
        # Check stopping criterion
        if  False not in stop_bool:
            stop = True
        #print("game value-",game_value)
        #print("allocation-",allocation)
        #print("stop",stop)

        misc = {
                "game_value": F_omega,
                "allocation": allocation,
                "optimal_policy": optimal_policy,
            }
        #print("misc-",misc)
        return arm, stop, optimal_policy, misc,pareto_arms, np.min(games), beta                
                

class Bandit:
    """
    Generic bandit class
    """

    def __init__(self, expected_rewards, sigma, seed):
        self.n_arms = expected_rewards.shape[1]
        self.expected_rewards = expected_rewards
        self.sigma = sigma
        self.seed = seed
        #self.random_state = np.random.multivariate_normal(expected_rewards,sigma,seed)

    def sample(self):
        pass

    def get_means(self):
        return self.expected_rewards


class GaussianBandit(Bandit):
    """
    Bandit with gaussian rewards
    """

    def __init__(self, expected_rewards, sigma, seed):
        super(GaussianBandit, self).__init__(expected_rewards, sigma, seed)
        self.sigma = sigma
        self.seed=seed

    def sample(self):
        out=[np.random.multivariate_normal(self.expected_rewards[i],cov=self.sigma,size=1) 
            for i in range(self.expected_rewards.shape[0])]
        return(np.array(out))


class BernoulliBandit(Bandit):
    """
    Bandit with bernoulli rewards
    """

    def __init__(self, expected_rewards, seed=None):
        super(BernoulliBandit, self).__init__(expected_rewards, seed)

    def sample(self):
        return self.random_state.binomial(1, self.expected_rewards)    


def run_exploration_experiment(bandit, explorer):
    """
    Run pure-exploration experiment for a given explorer and return stopping time and correctness
    """

    #optimal_policy = get_policy(bandit.get_means())
    done = False
    t = 0
    running_times = []
    #arm_count = np.array([0,0,0])
    pareto_arms_set = []
    while not done:
        t += 1
        # Act
        running_time = time.time()
        arm, done, policy, log, pareto_arms, game, beta= explorer.act()
        #print(policy)
        #arm_count[arm] += 1
        running_time = time.time() - running_time
        running_times.append(running_time)
        # Observe reward
        reward = bandit.sample()[arm]
        #print("reward - ",reward)
        # Update explorer
        explorer.update(arm, reward)
        
        if pareto_arms is not None:
            print(f"Pareto front - {pareto_arms} at time step {t} with minimum game value {game} and threshold {beta}")
            pareto_arms_set.append(pareto_arms)
        

        
    # Check correctness
    #correct = np.array_equal(optimal_policy, policy)

    # Return stopping time, correctness, optimal policy and recommended policy
    
    return t, policy, pareto_arms_set, running_times

def get_f1_score(p_t,p):
    f1_list = []
    for t in range(len(p_t)):
        tp = set(p).intersection(set(p_t[t]))
        fp = set(p_t[t]).difference(tp)
        fn = set(p).difference(tp)
        f1 = (2*len(tp))/(2*len(tp)+len(fp)+len(fn))
        f1_list.append(f1)
    return f1_list    

if __name__ == "__main__":
    #designs = np.array(pd.read_csv("~/Codes/preferece_pureexp/covboost.csv", delimiter=','))
    #print(designs)
    
    for rho in np.arange(-1,1,0.1):
        mean_stopping_times = []
        mu = np.array([[0.72875559,  1.20119222],
                        [0.45524805, -0.63317069],
                        [0.62826926,  1.27683777],
                        [0.94570734,  2.31592981],
                        [ 2.08131887,  1.4809387 ]])
        #print(mu.shape)
        #mu[:,0] = -mu[:,0]
        #theta_45 = np.pi/4
        #W_45_1 = np.array([-np.tan(np.pi/4-theta_45/2), 1])
        #W_45_2 = np.array([+np.tan(np.pi/4+theta_45/2), -1])
        #W_45_1 = W_45_1/np.linalg.norm(W_45_1)
        #W_45_2 = W_45_2/np.linalg.norm(W_45_2)
        #W_45 = np.vstack((W_45_1, W_45_2))

        W_90 = np.eye(mu.shape[1])

        #theta_135 = 3*np.pi/4
        #W_135_1 = np.array([-np.tan(np.pi/4-theta_135/2), 1])
        #W_135_2 = np.array([-np.tan(np.pi/4+theta_135/2), 1])
        #W_135_1 = W_135_1/np.linalg.norm(W_135_1)
        #W_135_2 = W_135_2/np.linalg.norm(W_135_2)
        #W_135 = np.vstack((W_135_1, W_135_2))
        #p_opt_45 = Vect_opt(mu, W_45)
        p_opt_90 = Vect_opt(mu, W_90)
        #p_opt_135 = Vect_opt(mu, W_135)
        #print(W_90)
        
        print(f"The Pareto front indices with the positive ortant preference cone are - {p_opt_90}")
        stopping_times = []
        iteration = 100
        delta = 0.01
        sigma = np.array([[1,rho],[rho,1]])
        for iter in range(iteration):
        
            explorer = FWS(n_arms = mu.shape[0], sigma=sigma, delta=delta, W = W_90, r =1)
            bandit = GaussianBandit(mu,sigma=sigma,seed = seed)
            
            #start = time.time()
            t, policy, pareto_arms_set, runtimes= run_exploration_experiment(bandit, explorer)
            #end = time.time()
            
            stopping_times.append(t)
            #f1_scores = get_f1_score(pareto_arms_set,p_opt_90)
            #print(f"Experiment {iter+1} stopped at {t}, Rec policy {policy} and Pareto front {pareto_arms}")
            '''
            with open(f'f1_score_APrePEx_rho_{delta}_{iteration}.txt', 'a') as f:
                f.write("\n")
                f.write(f"F1-scores in experiment {iter+1} - {f1_scores}")
                f.write("\n")
                f.close()
            
            with open(f'runtimes_APrePEx_rho_{delta}_{iteration}.txt', 'a') as f:
                f.write("\n")
                f.write(f"runtimes in experiment {iter+1} - {runtimes}")
                f.write("\n")
                f.close()
            '''
            print(f"{iter+1} done at {t}")
            
            
        mean_stopping_times.append(np.mean(stopping_times))
        
    with open(f'APrePEx_stopping_times_rho_{delta}_{iteration}.txt', 'a') as f:
        f.write("\n")
        f.write(f"Mean Stopping time is - {mean_stopping_times}")
        f.write("\n")
        f.close()
    

    


