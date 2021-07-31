import numpy as np
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import scipy.stats as ss

#%%
s0 = 49
mu = 0
k = 50
sigma = 0.2
q = 0
r = 0.05
T = 20 # weeks
numOfCall = 100000
timeBtwHedgeRebal = 1

#%%
def d1(s,k,r,q,sigma,tau):
    return (np.log(s/k) + (r-q+(sigma**2)/2)*tau) / (sigma*np.sqrt(tau))

def d2(s,k,r,q,sigma,tau):
    return d1(s,k,r,q,sigma,tau) - sigma*np.sqrt(tau)

def delta(s,k,r,q,sigma,tau):
    return ss.norm.cdf(d1(s,k,r,q,sigma,tau))

def optionPrice(s,k,r,q,sigma,tau):
    return s*np.exp(-q*tau)*ss.norm.cdf(d1(s,k,r,q,sigma,tau)) - k*np.exp(-r*tau)*ss.norm.cdf(d2(s,k,r,q,sigma,tau))

def hedgingCost(s0,mu,k,r,q,sigma,T,numOfCall,timeBtwHedgeRebal):
    ### week ###
    weeksToYears = np.arange(0,T+1,timeBtwHedgeRebal)/52
    
    ### stock path ###
    total_steps = len(weeksToYears)
    dt = timeBtwHedgeRebal/52
    
    s = np.zeros(total_steps-1)
    s = np.insert(s, 0, s0)
    for i in range(total_steps-1):
        z = np.random.randn(1)
        s[i+1] = s[i] * np.exp((mu-0.5*sigma**2)*dt + np.sqrt(dt)*(sigma*z))
    
    #s = np.array([49.00, 48.125, 47.375, 50.25, 51.75, 53.125, 53.00, 51.875, 51.375, 53.00, 49.875, 48.50, 49.875, 50.375, 52.125, 51.875, 52.875, 54.875, 54.625, 55.875, 57.25]) ### Text book example
    #s = np.array([49.00, 49.75, 52.0, 50.00, 48.38, 48.25, 48.75, 49.63, 48.25, 48.25, 51.12, 51.50, 49.88, 49.88, 48.75, 47.50, 48.00, 46.25, 48.13, 46.63, 48.12]) ### Text book example
    
    ### delta path ###
    delta_path = np.zeros(total_steps)
    for i in range(total_steps):
        delta_path[i] = delta(s[i],k,r,q,sigma,weeksToYears[len(weeksToYears)-(i+1)])
                
    ### Shares purchage ###
    sharesPurchased = np.zeros(total_steps)
    for i in range(total_steps):
        if i == 0:
            sharesPurchased[i] = delta_path[i] * numOfCall
        else:
            sharesPurchased[i] = (delta_path[i] - delta_path[i-1]) * numOfCall
     
    ### Cost of shares purchased ###
    costOfSharesPurchased = s * sharesPurchased
        
    ### Interest Cost & Cumulative cost including interest ###
    interestCost = np.zeros(total_steps)
    cumCostInclInterest = np.zeros(total_steps)
    for i in range(total_steps):
        if i == 0:
            cumCostInclInterest[i] = costOfSharesPurchased[i]
            interestCost[i] = cumCostInclInterest[i] * r * dt
    
        else:
            cumCostInclInterest[i] = cumCostInclInterest[i-1] + costOfSharesPurchased[i] + interestCost[i-1]
            
            if i == total_steps-1:
                interestCost[i] = 0.0
            else:
                interestCost[i] = cumCostInclInterest[i] * r * dt
    
    hedgingCost = cumCostInclInterest[-1] - k*numOfCall

    return np.round(hedgingCost,2)

#%%
print(hedgingCost(s0,mu,k,r,q,sigma,T,numOfCall,timeBtwHedgeRebal))
    