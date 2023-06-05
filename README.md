# A tool to analyze behavioral data in a temporal discounting study, with primary and secondary rewards

This tool is designed to
1. Convert log files from our conducted experiments into an easily machine-readable array format
2. For both conditions in the experiment (Primary and Secondary reward):
    1. For all five models  estimate the best fitting model parameters using both loglikelihood estimation and non-linear least squares
    2. calculate AIC and R<sup>2</sup> for all five temporal discounting equations discussed in Peters & Büchel 2012
Hyperbol without scaling, Exponential without scaling, Hyperbol with scaling of the delay, Hyperbol with scaling of the denominator, Exponential with scaling
3. Calculate ΔAIC for all conditions, subjects and models
4. Compare R2 and  ΔAIC for all models and conditions
5. Make images


|   | Hyperbol   | Exponential|
|---|---|---|
| Without scaling| $` SV={Amount \over 1+k*D} `$  | $` SV=A*exp(-kD) `$ |
| With scaling | $`SV={Amount \over (1+k*D)^s } `$ or $` SV={Amount \over 1+k*(D^s } `$   | $` SV=A*exp(-kD)^s `$  | 

## Install

## Log Files

## Description of functions

## Aim of analysis


