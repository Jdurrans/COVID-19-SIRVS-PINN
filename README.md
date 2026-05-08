# COVID-19 SIRS/SIRVS Physics informed nerual network

A PINN framework applied to the SIRS and SIRVS compartmental models to learn a time-varying transmission rate beta(t)
from English COVID-19 case data (2020-2022). This is then used to evaluate the impact of government interventions through 
quantifying Re(t) over the same time period.

#scripts
- "SIRS_SIRVS_PINN.py" - this is the main model, which trains the four wave-specific PINNs
- "x_sweep.py" - this sweeps across possible values for the data/physics loss weighting parameter x
- "consistency_check.py" - this validated the fixed epidemiological paramteres by keeping all other
                           paramters fixed and learning the paramter, then comparing it to the original fixed value

# requirements
pip install torch numpy matplotlib odfpy

#usage
- "SIRS_SIRVS_PINN.py" - (first run then subsequent runs once models are saved (change in paramteers require deleting original models and rerunning)
python SIRS_SIRVS_PINN.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90 --x 0.8 --ukhsa_r "ukhsa_r.ods"
python SIRS_SIRVS_PINN.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90 --x 0.8 --load_models --ukhsa_r "ukhsa_r.ods"
del model_wave1.pt; del model_wave2.pt; del model_wave3.pt; del model_wave4.pt

- "x_sweep.py" - (first run, and same line for subsequent runs, change in paramters require deleting the models first)
python x_sweep.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90
del model_wave1.pt; del model_wave2.pt; del model_wave3.pt; del model_wave4.pt

- "consistency_check.py" - (runs the entire code every time)
python consistency_check.py --data cases.csv --vax vax.csv --N 56000000 --I0 25000 --S0_frac 0.90
