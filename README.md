# Content 

This package permits to try bandit algorithms for piecewise stationary bandit problems with Bernoulli rewards and to reproduce experiments in the paper

* Efficient Change-Point Detection for Tackling Piecewise-Stationary Bandits by Lilian Besson, Emilie Kaufmann, Odalric-Ambrym Maillard and Julien Seznec, arXiv:1902.01575

# How does it work? 

To run bandit algorithms on piecewise stationary instances, you should custom the parameters of your experiment in main_NonStat.jl before running this file.   

All the fields that start with capital letters can be cumstomized. For example in # TYPE OF EXPERIMENT you specify whether you want to run experiment on a single instance (Frequentist) or on several randomly selected instance (Bayesian). Some benchmarks from the papers are proposed, but you can also custom your bandit instance.

The defaults parameters in main_NonStat.jl permit to run several algorithms on Problem 1 of [Besson et al. 20] up to a horizon T=20000, averaging results over N=100 independent repetitions. 

The defaults parameters in main_NonStat_Bayesian.jl permit to run experiments on N=100 different bandit instances obtained from the sampler described in [Besson et al. 20] up to a horizon T=20000 (besides the choice of parameters, this file is identical to main_NonStat.jl, so you can also use the latter to launch a Bayesian experiment). 

- choosing Save = "on" will save the results in the results folder 
- results (average final regret and its stdev, number of restarts) will be displayed in the command window anyways

If you have saved results from a Frequentist experiment, running view_results.jl will help visualizing them (regret curve + writing a CSV file with regret and number of restarts). 
Name and parameters, specified at the beginning of this file, should match with your saved data.

If you have saved results from a Bayesian experiment, running view_results_Bayesian.jl will help visualizing them (regret curve + writing a CSV file with regret and number of restarts). 
Name and parameters, specified at the beginning of this file, should match with your saved data.

Bandit algorithms are given in Algos.jl and useful functions to define piecewise stationary instances are given in Benchmarks.jl.  

# Configuration information

Experiments were run with the following Julia install: 

julia> VERSION
v"1.0.5"

(v1.0) pkg> status
 - CSV v0.7.7
 - DataFrames v0.21.8
 - Distributions v0.23.11
 - HDF5 v0.12.5
 - PyPlot v2.9.0

To install these package, you can run 

>using Pkg; Pkg.add("CSV"); Pkg.add("DataFrames"); Pkg.add("Distributions"); Pkg.add("HDF5"); Pkg.add("PyPlot")

# MIT License

Copyright (c) 2020 [Emilie Kaufmann]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
