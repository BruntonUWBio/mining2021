We used the [PyHSMM-Autoregressive](https://github.com/mattjj/pyhsmm-autoregressive) library to fit our 2-state discretization models.  
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

Full wrapper code snippet that fits the ARHMM:
```

```
