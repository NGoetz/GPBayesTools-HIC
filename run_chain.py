
from os import path
import os
import sys
sys.path.insert(0, path.abspath('../'))

from src.mcmc import Chain
from data_compress import parse_outname_and_generate_file
import pickle
import argparse

#two kinds of chains: one full with all data and compared to exp, one with validation set and compared to left out (1  left out)



def main():
    parser = argparse.ArgumentParser(description='Process training set files.')
    parser.add_argument('--training_set', nargs='+', required=True, help='List of training set files')
    args = parser.parse_args()

    training_set = args.training_set
    print(training_set)
    parent="../actual"
    
    model_par = '../training_points/configs/config_AuAu_200_bulk_scan_central.yaml'
    data_path = parent+"/latent_train_full/"
    closure_test=parent+"/latent_test/"
    for outname in training_set:
        # Remove the file extension to get the outname
        outname = outname.replace('.pkl', '')
        parse_outname_and_generate_file(data_path, outname, 0, 500, exp=True)
    output_file_list = []
    for file in training_set:
        name=file.split('/')[-1].split('.p')[0]
        output_file_list.append(name)


    import concurrent.futures

    def run_chain(tr_set, log=False, PCA=False, method='PCGP', closure=False):
        path_output_full=parent+'/emulator_full/'
        path_output_val=parent+'/emulator_val/'
        path_mcmc_full=parent+'/mcmc_full/'
        path_mcmc_val=parent+'/mcmc_val/'
        tag=""
        tag_clo=""
        if closure:
            tag_clo="closure"
            path_output = path_output_val
            path_mcmc = path_mcmc_val
        else:
            tag_clo="full"
            path_output = path_output_full
            path_mcmc = path_mcmc_full
    
        if log:
            tag=tag+"_log_"
            if PCA:
                tag=tag+"pca_"
                path_emu=path_output+'log/pca/'
                path_mcmc=path_mcmc+'log/pca/'
            else:
                tag=tag+"nopca_"
                path_emu=path_output+'log/no_pca/'
                path_mcmc=path_mcmc+'log/no_pca/'
        else:
            tag=tag+"_nolog_"
            if PCA:
                tag=tag+"pca_"
                path_emu=path_output+'no_log/pca/'
                path_mcmc=path_mcmc+'no_log/pca/'
            else:
                tag=tag+"nopca_"
                path_emu=path_output+'no_log/no_pca/'  
                path_mcmc=path_mcmc+'no_log/no_pca/'

        if method=='PCGP':
            path_emu=path_emu+'PCGP/'
            path_mcmc=path_mcmc+'PCGP/'
        else:
            path_emu=path_emu+'PCSK/'
            path_mcmc=path_mcmc+'PCSK/'

        if closure:
            parse_outname_and_generate_file(closure_test, tr_set.replace(".pkl",''), 499, 500, exp=False)
            exp_path = closure_test+ tr_set.replace(".pkl",'') +".pkl"
        else:
            parse_outname_and_generate_file("../exp/", tr_set.replace(".pkl",''), 499, 500, exp=True)
            exp_path = "../exp/"+ tr_set.replace(".pkl",'') +".pkl"

        path_emu = path_emu + tr_set.replace('.pkl', '') + tag + method

        mcmc_path = path_mcmc + tr_set.replace('.pkl', '') + tag + method + "_" + tag_clo+'.pkl'
        os.makedirs(path_mcmc, exist_ok=True)

        output_file = path_mcmc + tr_set.replace('.pkl', '') + tag + method + "_" + tag_clo + ".pkl"

        # Check if the output file already exists
        if not os.path.exists(output_file):
            mymcmc = Chain(mcmc_path=mcmc_path, expdata_path=exp_path, model_parafile=model_par)

            mymcmc.loadEmulator([path_emu])

            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["RDMAV_FORK_SAFE"] = "1"

            n_effective = 8000
            n_active = 4000
            n_prior = 32000
            sample = "tpcn"
            n_max_steps = 200
            random_state = 42

            n_total = 40000
            n_evidence = 40000

            pool = 40

            sampler = mymcmc.run_pocoMC(n_effective=n_effective, n_active=n_active,
                                        n_prior=n_prior, sample=sample,
                                        n_max_steps=n_max_steps, random_state=random_state,
                                        n_total=n_total, n_evidence=n_evidence, pool=pool)

            print("Run chain for ", tr_set, " with log_transform =", log, " and pca =", PCA, " and method =", method, " and closure =", closure)
            with open(output_file, "wb") as f:
                pickle.dump(sampler, f)
        else:
            print(f"MCMC result file {output_file} already exists. Skipping MCMC run.")

                                    
        
        print("Ran chain for ", tr_set, " with log_transform =", log, " and pca =", PCA, " and method =", method, " and closure =", closure)
        with open(path_mcmc + tr_set.replace('.pkl', '') + tag + method + "_" + tag_clo+".pkl", "rb") as f:
            chain = pickle.load(f)
            print(chain['chain'].shape)
            print(chain['log_likelihood'].shape)

    print("Start Chain")
    
    for tr_set in training_set:
        run_chain(tr_set, log=False, PCA=False, method='PCGP', closure=True)
        run_chain(tr_set, log=True, PCA=False, method='PCGP', closure=True)
        
        
        run_chain(tr_set, log=False, PCA=False, method='PCSK', closure=True)
        run_chain(tr_set, log=True, PCA=False, method='PCSK', closure=True)
    
        run_chain(tr_set, log=False, PCA=False, method='PCGP', closure=False)
        run_chain(tr_set, log=True, PCA=False, method='PCGP', closure=False)
        
        
        run_chain(tr_set, log=False, PCA=False, method='PCSK', closure=False)
        run_chain(tr_set, log=True, PCA=False, method='PCSK', closure=False)
        

        # run_chain(tr_set, log=False, PCA=True, method='PCSK', closure=True) 
        # run_chain(tr_set, log=True, PCA=True, method='PCSK', closure=True)
        #run_chain(tr_set, log=False, PCA=True, method='PCGP', closure=True)
        # run_chain(tr_set, log=True, PCA=True, method='PCGP', closure=False) #data:7.7_200:05:base:noetacut:nostarptcut_log_pca_PCGP_full.pkl this is trouble
        # run_chain(tr_set, log=False, PCA=True, method='PCGP', closure=False) #/actual/mcmc_full/no_log/pca/PCGP/data:7.7_200:05:base:noetacut:nostarptcut_nolog_pca_PCGP_full.pkl this ran through
        # run_chain(tr_set, log=False, PCA=True, method='PCSK', closure=False)
        # run_chain(tr_set, log=True, PCA=True, method='PCSK', closure=False)
        # run_chain(tr_set, log=True, PCA=True, method='PCGP', closure=True)
    print("End Chain")

if __name__ == "__main__":
    main()