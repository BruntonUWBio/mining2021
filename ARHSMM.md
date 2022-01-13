We used [Matt Johnson's](https://twitter.com/SingularMattrix) [PyHSMM-Autoregressive](https://github.com/mattjj/pyhsmm-autoregressive) library to fit our 2-state discretization models.  
* I recommend the [following example](https://github.com/mattjj/pyhsmm-autoregressive/blob/master/examples/demo.py) from the author   
* If you are having trouble installing PyHSMM, see [this page](https://github.com/mattjj/pyhsmm/issues/97)

The code snippet that fits the ARHMM in my code is as follows (longer version futher below):
```
posteriormodel = m.ARWeakLimitStickyHDPHMM(
        alpha=4., gamma=4., kappa=10., 
        init_state_distn='uniform',
        obs_distns=[
            d.AutoRegression(
                nu_0=D_latent+2,
                S_0=np.eye(D_latent),
                M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, D_latent*(nlags-1)+affine)))),
                K_0=np.eye(D_latent*nlags+affine),
                affine=affine)
            for state in range(Nmax)],
        )
```

**NOTE**: [PyHSMM-Autoregressive](https://github.com/mattjj/pyhsmm-autoregressive) is now quite out of date (package versioning issues etc) has been superceded by the excellent [SSM](https://github.com/lindermanlab/ssm) package which is actively maintained and documented by the [Liderman Lab](https://web.stanford.edu/~swl1/).


Full wrapper code snippet that fits the ARHMM:
```
import pyhsmm
from pyhsmm.util.text import progprint_xrange
from pyhsmm.util.stats import whiten, cov
import autoregressive.models as m
import autoregressive.distributions as d
from pyslds.util import get_empirical_ar_params
import pyslds
import numpy.random as npr
from pyslds.models import DefaultSLDS
from pyhsmm.util.text import progprint_xrange
import matplotlib.pyplot as plt
import subprocess 
import tqdm

def get_arhmm(data_array, nlags=1, affine=True, K_latent=2, N_samples=150, do_plot=False, _inference='gibbs'):
    print("Fitting AR-HMM")
    data_array_list = data_array # Work assuming a list of data-arrays has been passed in 
    if not isinstance(data_array_list, list):
        data_array_list = [data_array_list]
    T, D_obs = data_array_list[0].shape
    Nmax = K_latent 
    D_latent = D_obs

    print("ARWeakLimitStickyHDPHMM")
    posteriormodel = m.ARWeakLimitStickyHDPHMM(
        alpha=4., gamma=4., kappa=10., 
        # alpha=4., gamma=4., kappa=10., 
        init_state_distn='uniform',
        obs_distns=[
            d.AutoRegression(
                # nu_0=D_latent+1,
                # S_0=np.eye(D_latent),
                # M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, int(affine))))),
                # K_0=np.eye(D_latent + affine),
                
                nu_0=D_latent+2,
                S_0=np.eye(D_latent),
                M_0=np.hstack((np.eye(D_latent), np.zeros((D_latent, D_latent*(nlags-1)+affine)))),
                K_0=np.eye(D_latent*nlags+affine),

                affine=affine)
            for state in range(Nmax)],
        )

    for data_array in data_array_list:
        posteriormodel.add_data(data_array)

    posteriormodel, logliks = fit_model(posteriormodel, 
        N_samples=N_samples, do_plot=do_plot, _type=None, _inference=_inference)

    # only used for analysis of single videos, not a list of videos
    T = T - posteriormodel.nlags # fix for nlags; 
    
    return(posteriormodel, logliks, T)

def fit_model(posteriormodel, N_samples, do_plot=False, _type=None, _inference='gibbs'):
    print("Inference method:", _inference)
    print("posteriormodel:", posteriormodel)
    # Common 
    def gibbs_update(model):
        # print("Sampling...")
        model.resample_model() # TODO try: num_procs=4
        return model.log_likelihood()

    # Run the Gibbs sampler
    if _inference.lower() == 'gibbs':
        # logliks = [gibbs_update(posteriormodel) for _ in progprint_xrange(N_samples)]
        logliks = [gibbs_update(posteriormodel) for _ in tqdm.tqdm(range(N_samples))]

    return posteriormodel, logliks
```
