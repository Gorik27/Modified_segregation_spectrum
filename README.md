# Spectral model for grain boundary segregation in systems with strong solute-solute interactions
The source code for calculating the modified segregation spectrum incorporates pair solute-solute interactions that are valid for systems with strong solute-solute interactions (see original paper https://doi.org/10.1016/j.actamat.2025.121044 for details)

## Usage
The Python script ```modified_spectrum.py``` takes as input a set of energies of a polycrystal containing single solute atoms and pairs of solute atoms occupying grain boundaries (GBs). It also takes the energy of a polycrystal with no solute and a polycrystal with a single solute atom in the interior of a grain.
The script then calculates the segregation energies and pair interaction parameters, which are used to obtain a modified segregation energy spectrum.\
\
Feel free to contact us with any questions!

## Input files
All input and output files have to be in the subdirectory, which name is passed to the scipt through ```path``` variable (line 21).
### `GBEs.txt` - Energy of the polycrystall with single solutes

**format:**
>`id1 E1`\
> `id2 E2`\
> `...`

### `bulkEs.txt` - One or more energies of the polycrystal with the solute in the grain interior will be used. If there is more than one, the average will be used.

**format:**
>`id1 E1`\
> `id2 E2`\
> `...`

### `neigbors.txt` - List of neighbors IDs (for which solute-solute interactions is accounted for)

**format:**
>`id1 n1 neighbor_id11 neighbor_id12 .. neighbor_id1n1`\
>`id2 n2 neighbor_id21 neighbor_id22 .. neighbor_id2n2`\
> `...`

### `GBEs_int.txt` - Energies of the polycrystall filled with a pair of neihgboring solutes

**format:**
>`id1 E11 E12 ... E1n1`\
>`id2 E21 E22 ... E2n2`\
> `...`

### `pureE.txt` - Energy of the polycrystall without solutes

**format:**
>`E`

## Output files

### `modified_spectrum.txt` - The modified segregation spectrum

**format:**
>`E1 F1`\
>`E2 F2`\
>`...`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where `Fi` is a number of sites with energy `Ei` in i-th conglomerate

### `ids_c.txt` - IDs of atoms in conglomerates 

**format:**
>`id_1:1 id_1:2 ... id_1:F1`\
>`id_2:1 id_2:2 ... id_2:F2`\
>`...`

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; where `id_i:j` is the ID of j-th site in the i-th conglomerate

## Plotting 
Output data from the `modified_spectrum.txt` can be converted to a format suitable for histogram plotting using the following code
```
from itertools import chain, repeat
import numpy as np


data = np.loadtxt('modified_spectrum.txt')
E_s = data[:, 0]
F_s = data[:, 1].astype(int)

Ehist = list(chain.from_iterable(repeat(j, times = i) for i, j in zip(F_s, E_s)))
```

## Citation
When using please cite our paper  (pre-proof available now): \
G. Marchiy, D. Samsonov, and E. Mukhin, “Spectral model for grain boundary segregation in systems with strong solute-solute interactions,” Acta Materialia, May 2025, doi: 10.1016/j.actamat.2025.121044.
