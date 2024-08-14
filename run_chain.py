
from os import path
import os
import sys
sys.path.insert(0, path.abspath('../'))

from src.mcmc import Chain

exp_path = "../exp/exp.pkl"
model_par = '../training_points/configs/config_AuAu_200_bulk_scan_central.yaml'
mymcmc = Chain(expdata_path=exp_path, model_parafile=model_par)

folder = "../trained_emulators_no_PCA/"
emuPathList = [folder+"PCSK_trained.sav"
               ]
mymcmc.loadEmulator(emuPathList)

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["RDMAV_FORK_SAFE"] = "1"

n_effective=8000
n_active=4000
n_prior=16000
sample="tpcn"
n_max_steps=100
random_state=42

n_total = 30000
n_evidence = 30000

pool = 12

sampler = mymcmc.run_pocoMC(n_effective=n_effective, n_active=n_active,
                            n_prior=n_prior, sample=sample,
                            n_max_steps=n_max_steps, random_state=random_state,
                            n_total=n_total, n_evidence=n_evidence, pool=pool)
                            
import pickle
with open("./mcmc/chain.pkl", "rb") as f:
    chain = pickle.load(f)
    print(chain['chain'].shape)
    print(chain['log_likelihood'].shape)
