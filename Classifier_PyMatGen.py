#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon May 26 18:17:28 2025

@author: alpesh
"""


from mp_api.client import MPRester 
    

# Open and read the key from the file

#Get file path/name
filename = r"/Users/alpesh/Desktop/ML projects test/Test-ML-projects/key.txt"

def get_file_cotents(filename):
    try:
        with open(filename,'r') as f:
            return f.read().strip()
    except FileNotFoundError:
        print("'%s' file not found" %filename)

My_API= get_file_cotents(filename) #Alpesh_API

# print(My_API)#This will ensure to print my API

mpr=MPRester(My_API) #MPRester is a Python class 
                     #used to access data from the 
                     #Materials Project
                     #mpr is caller / demander assist
                     
# avlbl_field=mpr.materials.summary.available_fields
# print(avlbl_field) #Print Available Fields/Criteria

criteria = {'energy_above_hull' : {'$lte':0.02}, 'band_gap' : {'$gt':0}}
#criteria
props = ['formula_pretty','band_gap','density','formation_energy_per_atom','volume']
#field
# entries= mpr.summary.search(criteria)
# print(entries)


entries = mpr.materials.summary.search(
    energy_above_hull=(None, 0.02),  # max 0.02 (min is None = no lower bound)
    band_gap=(0, None),              # min 0 (max None = no upper bound)
    fields=props)


import pandas as pd

# Convert list of model objects to a pandas DataFrame by extracting 
# their dictionary representations
df_insulators = pd.DataFrame([entry.dict() for entry in entries])

# Check the keys of the first entry to understand structure
# Optional: see what keys are present
# print("Available fields:", df_insulator.columns.tolist())

avg_density_of_insulators = df_insulators['density'].mean()
std_density_of_insulators = df_insulators['density'].std()

print(f"Average density of insulators: {avg_density_of_insulators}")
print(f"Standard deviation of density: {std_density_of_insulators}")

entries = mpr.materials.summary.search(
    energy_above_hull=(None, 0.02),
    band_gap=(0, 1e-6),
    fields=props)

df_metals = pd.DataFrame([entry.dict() for entry in entries])

# Now filter strictly to band_gap == 0
df_metals = df_metals[df_metals['band_gap'] == 0]

avg_density_of_metals = df_metals['density'].mean()
std_density_of_metals = df_metals['density'].std()

print(f"Average density of metals: {avg_density_of_metals}")
print(f"Standard deviation of density: {std_density_of_metals}")



import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Assume df_insulators and df_metals are your dataframes from above

# Set style for better aesthetics
sns.set(style="whitegrid", palette="muted", font_scale=1.2)

# Create figure with 3 vertical subplots sharing y-axis (band gap)
fig, axes = plt.subplots(3, 1, figsize=(8, 15), sharey=False)
fig.subplots_adjust(hspace=0.35)

# Panel 1: Density
for ax, df, label, color in zip(axes, [df_insulators, df_metals], ['Insulators', 'Metals'], ['tab:blue', 'tab:orange']):
    pass  # we'll add data per panel below

# Density plot
ax = axes[0]
sns.kdeplot(df_insulators['density'], ax=ax, fill=True, label='Insulators', color='tab:blue')
sns.kdeplot(df_metals['density'], ax=ax, fill=True, label='Metals', color='tab:orange')
ax.set_xlabel('Density (g/cm³)')
ax.set_ylabel('Density Distribution')
ax.set_title('Density Distribution')
ax.legend()
x_min, x_max = min(df_insulators['density'].min(), df_metals['density'].min()), max(df_insulators['density'].max(), df_metals['density'].max())
ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))

# Volume plot
ax = axes[1]
sns.kdeplot(df_insulators['volume'], ax=ax, fill=True, label='Insulators', color='tab:blue')
sns.kdeplot(df_metals['volume'], ax=ax, fill=True, label='Metals', color='tab:orange')
ax.set_xlabel('Volume (Å³)')
ax.set_ylabel('Volume Distribution')
ax.set_title('Volume Distribution')
x_min, x_max = min(df_insulators['volume'].min(), df_metals['volume'].min()), max(df_insulators['volume'].max(), df_metals['volume'].max())
ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))

# Formation energy per atom (ΔH per atom) plot
ax = axes[2]
sns.kdeplot(df_insulators['formation_energy_per_atom'], ax=ax, fill=True, label='Insulators', color='tab:blue')
sns.kdeplot(df_metals['formation_energy_per_atom'], ax=ax, fill=True, label='Metals', color='tab:orange')
ax.set_xlabel('Formation Energy per Atom (eV)')
ax.set_ylabel('Distribution')
ax.set_title('Formation Energy per Atom Distribution')
x_min, x_max = min(df_insulators['formation_energy_per_atom'].min(), df_metals['formation_energy_per_atom'].min()), max(df_insulators['formation_energy_per_atom'].max(), df_metals['formation_energy_per_atom'].max())
ax.set_xlim(x_min - 0.1 * (x_max - x_min), x_max + 0.1 * (x_max - x_min))

plt.show()


import scipy.stats


# Define the properties of the mystery material
density = 8
volume = 60
formation_energy = -0.5
#is it a metal or insulator???

# Initial guess based on proportion of metals v insulators
prior_metals = df_metals['density'].count()/(df_insulators['density'].count()+df_metals['density'].count())
prior_insulators = 1-prior_metals
print('The first guess based on metal vs insulator proportion.')
print('Probability of being metal:',prior_metals)
print('Probability of being insulator:',prior_insulators,'\n')

# Probability based on density
density_metals = scipy.stats.norm(df_metals['density'].mean(), df_metals['density'].std()).pdf(density)
density_insulators = scipy.stats.norm(df_insulators['density'].mean(), df_insulators['density'].std()).pdf(density)
print('The second guess based on density.')
print('Density likelihood for metal:',density_metals)
print('Density likelihood for insulator:',density_insulators,'\n')

# Probability based on volume
volume_metals = scipy.stats.norm(df_metals['volume'].mean(), df_metals['volume'].std()).pdf(volume)
volume_insulators = scipy.stats.norm(df_insulators['volume'].mean(), df_insulators['volume'].std()).pdf(volume)
print('The third guess based on volume.')
print('Volume likelihood for metal:',volume_metals)
print('Volume likelihood for insulator:',volume_insulators,'\n')

# Probability based on formation energy
energy_metals = scipy.stats.norm(df_metals['formation_energy_per_atom'].mean(), df_metals['formation_energy_per_atom'].std()).pdf(formation_energy)
energy_insulators = scipy.stats.norm(df_insulators['formation_energy_per_atom'].mean(), df_insulators['formation_energy_per_atom'].std()).pdf(formation_energy)
print('The Fourth guess based on formation energy.')
print('Energy likelihood for metal:',energy_metals)
print('Energy likelihood for insulator:',energy_insulators,'\n')

# Now we add up the log of these probabilities and compare
odds_of_metal = np.log(prior_metals)+np.log(density_metals)+np.log(volume_metals)+np.log(energy_metals)
odds_of_insulator = np.log(prior_insulators)+np.log(density_insulators)+np.log(volume_insulators)+np.log(energy_insulators)
print('Our final guess is based on all of these probabilities combined!')
print('The odds of being a metal are:',odds_of_metal)
print('The odds of being an insulator are:',odds_of_insulator,'\n')

# Classify the material using the found odds
if odds_of_metal > odds_of_insulator:
    print('new material is probably a metal!')
else:
    print('new material is an insulator!')

    
