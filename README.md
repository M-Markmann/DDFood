# A tool to analyze behavioral data in a temporal discounting study, with primary and secondary rewards

This tool is designed to
1. Convert log files from our conducted experiments into an easily machine-readable array format
2. For both conditions in the experiment (Primary and Secondary reward):
    1. For all five models  estimate the best fitting model parameters using both loglikelihood estimation and non-linear least squares
    2. calculate AIC and R<sup>2</sup> for all five temporal discounting equations discussed in Peters & Büchel 2012
Hyperbol without scaling, Exponential without scaling, Hyperbol with scaling of the delay, Hyperbol with scaling of the denominator, Exponential with scaling
3. Calculate ΔAIC for all conditions, subjects and models
4. Compare R2 and  ΔAIC for all models and conditions
5. Create plots


|   | Hyperbol   | Exponential|
|---|---|---|
| Without scaling| $` SV={Amount \over 1+k*D} `$  | $` SV=A*exp(-kD) `$ |
| With scaling | $`SV={Amount \over (1+k*D)^s } `$ or $` SV={Amount \over 1+k*(D^s } `$   | $` SV=A*exp(-kD)^s `$  | 

## Install

This tool uses the following libraries and versions:
-scipy 1.9.3
-seaborn 0.11.2
-numpy 1.23.4
-matplotlib 3.3.1
-pandas 1.4.2
-python 3.8.10





