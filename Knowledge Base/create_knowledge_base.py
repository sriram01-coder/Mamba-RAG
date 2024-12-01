import datasets
from datasets import load_dataset
from tqdm import tqdm
import collections
import os
import numpy as np
import torch
import pandas as pd
from collections import Counter
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

target_articles = ['API gravity', 'Amplitude', 'Angular momentum', 'Antiferromagnetism', 'Astrochemistry', 'Baryogenesis', 'Black hole', 'Bollard pull', 'Born reciprocity', 'Butterfly effect', 'C1 chemistry', 'Causality (physics)', 'Cavitation', 'Clockwise', 'Coffee ring effect', 'Coherence (physics)', 'Coherent turbulent structure', 'Cold dark matter', "Commentary on Anatomy in Avicenna's Canon", 'Condensation cloud', 'Convection (heat transfer)', 'Copernican principle', 'Critical Raw Materials Act', 'Crossover experiment (chemistry)', 'Crystallinity', 'Dark Matter', 'Decay technique', 'Diffraction', 'Dimension', 'Droste effect', 'Dynamic scaling', "Earnshaw's theorem", 'Ecological pyramid', 'Electric flux', 'Electrical resistivity and conductivity', 'Electrochemical gradient', 'Electronic entropy', "Elitzur's theorem", 'Emissivity', 'Enthalpy', 'Environmental Science Center', 'Erlangen program', 'Explicit symmetry breaking', "Fermat's principle", 'Ferromagnetism', 'Frame-dragging', 'Free neutron decay', 'Galaxy', 'Geometric quantization', 'Gravitational wave', 'Gravity Probe B', 'Heart', 'Heat treating', "Hesse's principle of transfer", 'History of geology', 'Hydrodynamic stability', 'Improper rotation', 'Infectious tolerance', 'Inflation (cosmology)', 'Interstellar medium', 'James Webb Space Telescope', 'Kutta-Joukowski theorem', 'Landau–Lifshitz–Gilbert equation', 'Leidenfrost effect', 'Light-year', 'Linear time-invariant system', 'List of equations in classical mechanics', 'Lorentz covariance', 'Luminance', 'Magnetic monopole', 'Magnetic resonance imaging', 'Magnetic susceptibility', 'Magnitude (astronomy)', 'Main sequence', 'Mammary gland', 'Mass versus weight', 'Mass-to-charge ratio', 'Memristor', 'Minkowski space', 'Modified Newtonian dynamics', 'Molecular cloud', 'Molecular symmetry', 'Morphology (biology)', 'Navier–Stokes equations', 'Nebula', "Newton's law of universal gravitation", 'Nuclear fusion', 'Observable universe', 'Organography', 'Paramagnetism', 'Parity (physics)', 'Phageome', 'Phase transition', 'Photophoresis', 'Planetary system', 'Plant', 'Point groups in three dimensions', 'Probability amplitude', 'Probability density function', 'Propagation constant', 'Pulsar', 'Pulsar-based navigation', 'QCD matter', 'Quantization (physics)', 'Quartz crystal microbalance', 'Radiosity (radiometry)', 'Ramsauer–Townsend effect', 'Rayleigh scattering', 'Reciprocal length', 'Redshift', 'Refractive index', 'Regular polytope', 'Relative density', 'Renormalization', 'Ring-imaging Cherenkov detector', 'Scale (ratio)', 'Second law of thermodynamics', 'Self-organization in cybernetics', 'Shower-curtain effect', 'Signal-to-noise ratio', 'Spatial dispersion', 'Speed of light', 'Spin (physics)', 'Spontaneous symmetry breaking', 'Standard Model', 'Stellar classification', 'Stochastic differential equation', 'Superconductivity', 'Supermassive black hole', 'Supernova', 'Supersymmetric quantum mechanics', 'Supersymmetry', 'Surface power density', 'Surgical pathology', 'Symmetry in biology', 'Symmetry of diatomic molecules', 'The Ambidextrous Universe', 'Theorem', 'Theorem of three moments', 'Thermal equilibrium', 'Tidal force', 'Time', 'Time standard', 'Total internal reflection', 'Triskelion', 'Ultraviolet catastrophe', 'Unified field theory', 'Uniform tilings in hyperbolic plane', 'Vacuum', 'Virtual particle', 'Water hammer', 'Wigner quasiprobability distribution', 'Work function', 'Zero-point energy']

dataset = load_dataset(f"Cohere/wikipedia-22-12-en-embeddings", split="train")

first_paraphs = dataset.filter(lambda example: example["paragraph_id"]==0, num_proc=20)

titles = dataset["title"]
first_paraph_titles = first_paraphs["title"]

docs_titles = first_paraphs["title"]
docs_embeddings = first_paraphs["emb"]

doc_embeddings_array = np.array(docs_embeddings)


kmean_dict = {}
for num_clusters in [20,]:
    kmean_dict[num_clusters] = KMeans(n_clusters=num_clusters,
                                      random_state=0,
                                      n_init=1).fit(doc_embeddings_array)

cluster_numbers = kmean_dict[20].predict(doc_embeddings_array)

cluster_dict = collections.defaultdict(list)
for dx,cl_id in enumerate(cluster_numbers):
    cluster_dict[cl_id].append(first_paraph_titles[dx])

sorted([(cluster_numbers[dx],i) for dx,i in enumerate(first_paraph_titles) if i in target_articles])

filtered_articles = cluster_dict[13]
filtered_articles_set = set(filtered_articles)

first_paraphs_filtered = first_paraphs.filter(lambda example: example["title"] in filtered_articles_set,
                                              num_proc=20)

docs_titles_filtered = first_paraphs_filtered["title"]
docs_embeddings_filtered = first_paraphs_filtered["emb"]

doc_embeddings_array_filtered = np.array(docs_embeddings_filtered)

kmean_dict_filtered = {}
for num_clusters in tqdm([6]):
    kmean_dict_filtered[num_clusters] = KMeans(n_clusters=num_clusters,
                                               random_state=0,
                                               n_init=1).fit(doc_embeddings_array_filtered)
    
cluster_numbers_filtered = kmean_dict_filtered[6].predict(doc_embeddings_array_filtered)

cluster_dict = collections.defaultdict(list)
for dx,cl_id in enumerate(cluster_numbers_filtered):
    cluster_dict[cl_id].append(docs_titles_filtered[dx])

sorted([(cluster_numbers_filtered[dx],i) for dx,i in enumerate(docs_titles_filtered) if i in target_articles])

filtered_articles1 = cluster_dict[0] + cluster_dict[1] + cluster_dict[2] + cluster_dict[4] + cluster_dict[5] #277046 articles
filtered_articles_set1 = set(filtered_articles1)

first_paraphs_filtered1 = first_paraphs_filtered.filter(lambda example: example["title"] in filtered_articles_set1,
                                                        num_proc=20)

docs_titles_filtered1 = first_paraphs_filtered1["title"]
docs_embeddings_filtered1 = first_paraphs_filtered1["emb"]

dataset_filtered = dataset.filter(lambda example: example["title"] in filtered_articles_set1,
                                                num_proc=20)

dataset_filtered.remove_columns(["emb"]).save_to_disk("/data6/sriramd/DLNLP/wikipedia.hf")