import numpy as np
from matplotlib import pyplot as plt
from copy import deepcopy as dc
from numba import njit, prange
import numba as nb
from itertools import chain, repeat

@nb.njit(parallel=True)
def isin(a, b):
    out=np.empty(a.shape, dtype=nb.boolean).ravel()
    ar = a.ravel()
    b = set(b)
    for i in prange(out.shape[0]):
        if ar[i] in b:
            out[i]=True
        else:
            out[i]=False
    return out.reshape(a.shape)

eV2kJmol = 96.485
path = 'Ag(Ni)'

data = np.loadtxt(f'{path}/GBEs.txt')
data = data[data[:, 0]!=0]
ids = data[:, 0]
Es = data[:, 1]
data = np.loadtxt(f'{path}/bulkEs.txt')
if len(data.shape)==1:
    E0 = data[1]
else:
    E0 = np.mean(data[:,1])

Eseg0 = (Es-E0)*eV2kJmol
srt1 = np.argsort(ids)
Eseg = Eseg0[srt1]
ids = ids[srt1]
srtE = np.argsort(Eseg)
plt.hist(Eseg[srtE],density=True, bins=40)
plt.xlabel('$\Delta E_i^{seg}$, kJ/mol')
plt.ylabel('probability density')
plt.gca().set_yticklabels([])
plt.show()

print('reading neighbor list...')
data = np.loadtxt(f'{path}/GBEs_int.txt')
len0 = len(data)
ids_n = np.zeros((len0, 2*(data.shape[1]-1)))
ids_c = np.zeros(len0)
where_repeat = -np.ones(len0).astype(int)
with open(f'{path}/neigbors.txt', 'r') as f:
    for i, line in enumerate(f):
        if i > 0 and i-1 < len(ids_c):
            df = line.replace('\n', '').split(' ')
            ids_c[i-1] = int(df[0]) 
            if int(df[1])>0:
                df.remove('')
                
                neighbor_list = np.array(df[2:]).astype(int) 
                
                where = np.where(neighbor_list==int(df[0]))
                if len(where[0])>0:
                    where_repeat[i-1] = where[0][0]
                neighbor_list = neighbor_list[neighbor_list!=int(df[0])]
                ids_n[i-1, :len(neighbor_list)] = neighbor_list[:]

msk = (data[:, 0]!=0)
data = data[msk]
data = data[:, 1:]
for i in range(len(data)):
    if where_repeat[i]!=-1:
        data[i, where_repeat[i]:-1] = data[i, where_repeat[i]+1:]
        data[i, -1] = 0
Eint = np.zeros((len0, data.shape[1]*2))
Eint[:len(data), :data.shape[1]] = data[:, :]*eV2kJmol

print('done')
print('recovering of discarded part of Eint...')
ids_nc = dc(ids_n)
for i in range(len(ids_nc)):
    for j in range(len(ids_nc[i])):
        e = ids_nc[i][j]
        
        if e != 0:
            i1 = np.where(ids_c==e)[0][0]
            if len(ids_n[i1])>0:
                j1 = np.where(ids_n[i1]==0)[0][0]
            else:
                j1 = 0
            assert ids_n[i1][j1] == 0
            assert ids_c[i1] == e
            assert Eint[i1][j1] == 0
            ids_n[i1][j1] = ids_c[i]
            Eint[i1][j1] = Eint[i][j]
        else:
            break
    
zmax = np.sum(ids_n!=0, axis=1).max()
ids_n = ids_n[:, :zmax]
Eint = Eint[:, :zmax]

srt2 = np.argsort(ids_c)
assert np.all(ids_c[srt2]==ids)
ids_c = ids_c[srt2]
ids_n = ids_n[srt2]
Eint = Eint[srt2]

print('done')     
Epure = float(np.loadtxt(f'{path}/pureE.txt'))*eV2kJmol

w = Eint +  Epure - 2*E0*eV2kJmol
w[Eint==0] = 0
print('calculation of w...')
for i in range(len(w)):
    for j in range(len(w[i])):
        if w[i][j]!=0:
            j1 = np.where(ids==ids_n[i][j])[0][0]
            w[i][j] -= (Eseg[i]+Eseg[j1])
w[Eint==0] = 0
print('done')

wavg = np.sum(w, axis = 1)
corr = np.mean(np.corrcoef(wavg,Eseg)[0,1])
print('correlation between wavg and Eseg: ',  corr)

#%%
Em = np.mean(Eseg)
wm = np.mean(wavg)
mean = np.array([Em, wm])
cov = np.cov([Eseg, wavg])
print(f'mean: {mean}')
print(f'cov: {cov}')

#%%
plt.hist(w[w!=0], bins=100)
plt.xlabel('$\omega_{ij}$, kJ/mol')
plt.ylabel('probability density')
plt.gca().set_yticklabels([])
plt.show()

plt.hist(wavg, bins=100)
plt.xlabel('$\Omega_{i, gb}$, kJ/mol')
plt.ylabel('probability density')
plt.gca().set_yticklabels([])
plt.show()

plt.plot(Eseg, wavg, '.')
plt.xlabel('$\Delta E_i^{seg}$, kJ/mol')
plt.ylabel('$\Omega_{i, gb}$, kJ/mol')
plt.show()

#%%

plt.figure(dpi=800)
a = 0.8
dy = 1
nbins = round((w.max()-w.min())/dy)
density = False
wt = 0.1

_, _, bars = plt.hist(w[w!=0], bins=nbins, density=density, alpha=1,
                        label='1 neighbor shell', edgecolor='black',
                        linewidth=wt, color='cadetblue')

[bar.set_height(bar.get_height()/len(w)) for bar in bars]

plt.ylim((0, 1.1*np.max([bar.get_height() for bar in bars])))

plt.xlim((-30, 30))    
kT = 2*0.025*eV2kJmol*300/300
plt.axvline(-kT, color='dimgrey', label='$\mathrm{\pm 2kT, 300K}$')
plt.axvline(kT, color='dimgrey')
plt.xlabel('interaction parameter $\mathrm{\omega_{ij}, kJ/mol}$')
plt.ylabel('average number of interactions per atom $\mathrm{\overline{z}_{\omega}}$')
plt.legend(loc='upper right')

prec = 0

sz = 13
_za = (w<-kT).sum()/len(w)
za = round(_za, prec)
if prec == 0:
    za = int(za)
plt.annotate('$\mathrm{\overline{z}^{1\\ shell}_{\omega<-2kT}} = $'+f'{za}', 
              (-10.5, 0.6), xytext=(-26.5, 0.8), size=sz,
              arrowprops=dict(facecolor='black', 
                              width=0.1, headwidth=5, headlength=3))
_zr = (w>kT).sum()/len(w)
zr = round(_zr, prec)
if prec == 0:
    zr = int(zr)
plt.annotate('$\mathrm{\overline{z}^{1\\ shell}_{\omega>2kT}} = $'+f'{zr}', 
              (7, 0.15), xytext=(17, 0.5), size=sz,
              arrowprops=dict(facecolor='black', 
                              width=0.1, headwidth=5, headlength=3))


plt.show()
#%%
"""
simple ordering
"""

Eselected = dc(Eseg)
wselected = dc(w)
ids_c_selected = dc(ids_c)
ids_n_selected = dc(ids_n)

Er = dc(Eseg)
print('ordering...')

@njit
def renorm(Er, wselected, Eselected, ids_c_selected, ids_n_selected):
    for i in range(0, len(Er)):
        if i>0:
            ids_filled = ids_c_selected[i-1]
            msk = (ids_n_selected[i:]==ids_filled)
            O = np.sum(wselected[i:]*msk, axis=1)
        else:
            O = np.sum(wselected[i:]*np.zeros_like(wselected), axis=1)
        Eselected[i:] = Eselected[i:] + O
        i0 = np.argmin(Eselected[i:])
        t0 = Eselected[i:][i0]
        Er[i] = t0
        i0 = i+i0 
        wselected[i0], wselected[i] = wselected[i].copy(), wselected[i0].copy()
        Eselected[i0], Eselected[i] = Eselected[i], Eselected[i0]
        ids_c_selected[i0], ids_c_selected[i] = ids_c_selected[i], ids_c_selected[i0]
        ids_n_selected[i0], ids_n_selected[i] = ids_n_selected[i].copy(), ids_n_selected[i0].copy()
    return
    

renorm(Er, wselected, Eselected, ids_c_selected, ids_n_selected)
#%%
"""
conglomerate ordering
"""
plot_each = 5000

E_s = dc(Er)
F_s = np.ones(Er.shape, dtype=int)
w_n = dc(wselected) 


ids_c_s = -np.ones((len(Er), len(Er)), dtype=int)
ids_c_s[:, 0] = ids_c_selected

@njit
def conglomerate_interaction(ids_c_s, i0):
    O = 0
    lst = ids_c_s[i0-1]
    lst = lst[lst!=-1]
    for ind in lst:
        pind = np.where(ids_c_selected==ind)
        mask = isin(ids_n_selected[pind], ids_c_s[i0]) # is neighbors of [i-1] the member of conglomerate [i]?
        O += np.sum(w_n[pind]*mask) # bonds with current site
    return O

cnt = 0
@njit
def permutation(E_s, F_s, ids_c_s, cnt):
    change_flag = False
    for i in range(len(Er)-1-cnt, 0, -1): # reverse order
        if E_s[i] < E_s[i-1]:
            change_flag = True
            
            O = conglomerate_interaction(ids_c_s, i)
            
            if E_s[i] - O/F_s[i] < E_s[i-1] + O/F_s[i-1]: # case when solutes does not form a cluster
                # replace it with corresponding changes in bonds energy
                
                # E_s
                t = E_s[i]
                E_s[i] = E_s[i-1] + O/F_s[i-1]
                E_s[i-1] = t - O/F_s[i]
                
                # F_s
                t = F_s[i]
                F_s[i] = F_s[i-1]
                F_s[i-1] = t
                
                # ids_c_s
                t = ids_c_s[i].copy()
                ids_c_s[i] = ids_c_s[i-1]
                ids_c_s[i-1] = t
                
            else: # solutes form a cluster
                E2 = (E_s[i-1]*F_s[i-1] + E_s[i]*F_s[i])/(F_s[i-1]+F_s[i])
                cnt += 1
                # combine elements into one and shift right side of array
                
                # Combine
                E_s[i-1] = E2 # E_s
                F_s[i-1] += F_s[i] # F_s
                
                # ids_c_s
                ids3 = np.array(list(set(ids_c_s[i-1]).union(set(ids_c_s[i]))))
                ids3 = ids3[ids3!=-1]
                ids_c_s[i-1, :len(ids3)] = ids3 
                
                # Shift
                
                # E_s
                E_s[i:] = np.roll(E_s[i:], -1)
                E_s[-1] = 0
                
                # F_s
                F_s[i:] = np.roll(F_s[i:], -1)
                F_s[-1] = 0
                
                # ids_c_s
                for j in range(i, len(ids_c_s)-1):
                    ids_c_s[j] = ids_c_s[j+1]
                ids_c_s[-1] = -np.ones(len(Er))
                break
        elif change_flag:
            i0 = i
            break
    return i0, change_flag, cnt
    
change_flag = True
iteration = 0
i0 = len(Er)
print('conglomerate ordering...')

while change_flag:
    iteration += 1
    i0, change_flag, cnt = permutation(E_s, F_s, ids_c_s, cnt)
    if iteration % plot_each == 0:
        print(f'iteration #{iteration}, last step at {i0}/{len(Er)} site')
        plt.plot(E_s)
        plt.xlabel('site number')
        plt.ylabel('seg. energy')
        plt.title('conglomerate ordering')
        #plt.savefig(f'plots/plot_{iteration}.png')
        plt.show()
print('done')
#%%
Ehist = list(chain.from_iterable(repeat(j, times = i) for i, j in zip(F_s, E_s)))
Fhist = list(chain.from_iterable(repeat(j, times = i) for i, j in zip(F_s, F_s)))
plt.hist(Ehist, bins=50, density=True, alpha=1, label='modified spectrum')
plt.hist(Eseg, bins=50, alpha=0.4, density=True, label='original spectrum')
plt.xlabel('$E_{seg}$')
plt.ylabel('probability density')
plt.gca().set_yticklabels([])
plt.legend()
plt.show()

msk = F_s!=0
out = np.array([E_s[msk], F_s[msk]]).transpose()
np.savetxt(f'{path}/modified_spectrum.txt', out, header='E F')
out = dc(ids_c_s)
N = (out[:,0]!=-1).sum()
out = out[:N, :]
cmax = (out!=-1).sum(axis=1).max()
out = out[:, :cmax]
np.savetxt(f'{path}/ids_c.txt', out)
