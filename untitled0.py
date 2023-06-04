import random
import numpy as np
import matplotlib.pyplot as plt


P_b0_given_a = {'a-': 0.7, 'a+': 0.2}
P_b1_given_a = {'a-': 0.3, 'a+': 0.8}
P_c0_given_a = {'a-': 0.5, 'a+': 0.7}
P_c1_given_a = {'a-': 0.5, 'a+': 0.3}
P_d0_given_a = {'a-': 0.3, 'a+': 0.8}
P_d1_given_a = {'a-': 0.7, 'a+': 0.2}
P_g0_given_a = {'a-': 0.4, 'a+': 0.6}
P_g1_given_a = {'a-': 0.6, 'a+': 0.4}
P_f0_given_d = {'d-': 0.8, 'd+': 0.3}
P_f1_given_d = {'d-': 0.2, 'd+': 0.7}
P_a0 = 0.3
P_a1 = 0.7
P_e0_given_a_b = {'a-, b-': 0.6, 'a-, b+': 0.5, 'a+, b-': 0.3, 'a+, b+': 0.8}
P_e1_given_a_b = {'a-, b-': 0.4, 'a-, b+': 0.5, 'a+, b-': 0.7, 'a+, b+': 0.2}

elim_meth = 0.6428

def sample(probability, variable):
    
    rand_num = random.random()
    if rand_num < probability:
        return variable + '-'
    else:
        return variable + '+'
    
    
    
def rejection_sampling(num_samples):
    
    cnt_e0_given_f1 = 0
    cnt_f1 = 0
    
    for i in range(num_samples):
            
        a = 'a-' if random.random() < P_a0 else 'a+'
        b = sample(P_b0_given_a[a], 'b')
        d = sample(P_d0_given_a[a], 'd')
        f = sample(P_f0_given_d[d], 'f')
        if f == 'f+':
            cnt_f1 += 1
        else:
            continue
        e = sample(P_e0_given_a_b[f"{a}, {b}"], 'e')
        if e == 'e-':
            cnt_e0_given_f1 += 1
            
    estimate = cnt_e0_given_f1/cnt_f1
    return estimate

def likelihood_weighting(num_samples):
    
    weight_e0_given_f1 = 0
    weight_e1_given_f1 = 0
    
    for i in range(num_samples):
        
        weight = 1
        a = 'a-' if random.random() < P_a0 else 'a+'
        b = sample(P_b0_given_a[a], 'b')
        d = sample(P_d0_given_a[a], 'd')
        weight *= P_f1_given_d[d]
        e = sample(P_e0_given_a_b[f"{a}, {b}"], 'e')
        if e == 'e-':
            weight_e0_given_f1 += weight 
        else:
            weight_e1_given_f1 += weight
        
    estimate = weight_e0_given_f1/(weight_e0_given_f1 + weight_e1_given_f1)
    return estimate

def gibbs_sampling(num_samples):
    
    a = 'a-' if random.random() < P_a0 else 'a+'
    b = sample(P_b0_given_a[a], 'b')
    c = sample(P_c0_given_a[a], 'c')
    d = sample(P_d0_given_a[a], 'd')
    e = sample(P_e0_given_a_b[f"{a}, {b}"], 'e')
    g = sample(P_g0_given_a[a], 'g')
    states = {'a':a, 'b':b, 'c':c, 'd':d, 'e':e, 'f':'f+', 'g':g}
    cnt_eplus = 0
    cnt_eminus = 0
    for i in range(num_samples):
        #a
        P_b = P_b0_given_a if states['b'] == 'b-' else P_b1_given_a
        P_c = P_c0_given_a if states['c'] == 'c-' else P_c1_given_a
        P_e = P_e0_given_a_b if states['e'] == 'e-' else P_e1_given_a_b
        P_d = P_d0_given_a if states['d'] == 'd-' else P_d1_given_a
        P_g = P_g0_given_a if states['g'] == 'g-' else P_g1_given_a
        aminus = P_a0*P_b['a-']*P_e[f"a-, {states['b']}"]*P_d['a-']
        aplus = P_a1*P_b['a+']*P_e[f"a+, {states['b']}"]*P_d['a+']
        P_a0_given_edb = aminus/(aminus + aplus)
        a = sample(P_a0_given_edb, 'a')
        states['a'] = a
        if states['e'] == 'e-':
            cnt_eminus += 1
        else:
            cnt_eplus += 1
        #b
        P_a = 0.3 if states['a'] == 'a-' else 0.7
        P_e = P_e0_given_a_b if states['e'] == 'e-' else P_e1_given_a_b
        bminus = P_a*P_b0_given_a[states['a']]*P_e[f"{states['a']}, b-"]
        bplus = P_a*P_b1_given_a[states['a']]*P_e[f"{states['a']}, b+"]
        P_b0_given_ae = bminus/(bminus + bplus)
        b = sample(P_b0_given_ae, 'b')
        states['b'] = b
        if states['e'] == 'e-':
            cnt_eminus += 1
        else:
            cnt_eplus += 1
        #d
        P_a = 0.3 if states['a'] == 'a-' else 0.7
        dminus = P_a*P_d0_given_a[states['a']]*P_f1_given_d['d-']
        dplus = P_a*P_d1_given_a[states['a']]*P_f1_given_d['d+']
        P_d0_given_af = dminus/(dminus + dplus)
        d = sample(P_d0_given_af, 'd')
        states['d'] = d
        if states['e'] == 'e-':
            cnt_eminus += 1
        else:
            cnt_eplus += 1
        #e
        P_a = 0.3 if states['a'] == 'a-' else 0.7
        P_b = P_b0_given_a if states['b'] == b else P_b1_given_a
        eminus = P_a*P_b[states['a']]*P_e0_given_a_b[f"{states['a']}, {states['b']}"]
        eplus = P_a*P_b[states['a']]*P_e1_given_a_b[f"{states['a']}, {states['b']}"]
        P_e0_given_ab = eminus/(eminus + eplus)
        e = sample(P_e0_given_ab, 'e')
        states['e'] = e
        if states['e'] == 'e-':
            cnt_eminus += 1
        else:
            cnt_eplus += 1
    
    estimate = cnt_eminus/(cnt_eminus + cnt_eplus)
    return estimate               
        
        

Nr = 100
N = 10000

plt.figure(figsize=(5,12))

rejection_estimates = []
for i in range(Nr):
    estimate = rejection_sampling(N)
    rejection_estimates.append(estimate)

rejection_mean = np.mean(rejection_estimates)
rejection_std = np.std(rejection_estimates)
print("rs mean", rejection_mean)
print("rs std", rejection_std)
plt.subplot(3, 1, 1)
plt.hist(rejection_estimates)
plt.axvline(elim_meth, color='r', linestyle='--')
plt.title('rejection')
plt.axvline(rejection_mean, color='g')
plt.axvline(rejection_mean + rejection_std, color='g', linestyle='--')
plt.axvline(rejection_mean - rejection_std, color='g', linestyle='--')


likelihood_estimates = []
for i in range(Nr):
    estimate = likelihood_weighting(N)
    likelihood_estimates.append(estimate)

likelihood_mean = np.mean(likelihood_estimates)
likelihood_std = np.std(likelihood_estimates)
print("lw mean:", likelihood_mean)
print("lw std", likelihood_std)
plt.subplot(3, 1, 2)
plt.hist(likelihood_estimates)
plt.axvline(elim_meth, color='r', linestyle='--')
plt.axvline(likelihood_mean, color='g')
plt.axvline(likelihood_mean + likelihood_std, color='g', linestyle='--')
plt.axvline(likelihood_mean - likelihood_std, color='g', linestyle='--')
plt.title('likelihood')

gibbs_estimates = []
for i in range(Nr):
    estimate = gibbs_sampling(N)
    gibbs_estimates.append(estimate)

gibbs_mean = np.mean(gibbs_estimates)
gibbs_std = np.std(gibbs_estimates)
print("gs mean", gibbs_mean)
print("gs std", gibbs_std)
plt.subplot(3, 1, 3)
plt.hist(gibbs_estimates)
plt.axvline(elim_meth, color='r', linestyle='--')
plt.axvline(gibbs_mean, color='g')
plt.axvline(gibbs_mean + gibbs_std, color='g', linestyle='--')
plt.axvline(gibbs_mean - gibbs_std, color='g', linestyle='--')
plt.title('gibbs')

plt.show()












