# Visualize results of Bayesian experiments (average over many different problems) that are saved in the results folder (names should match)

using Statistics
using PyPlot
using HDF5
using DataFrames
using CSV

# OPTIONAL: write results in a csv file
updateCSV = "on" 


# NAME OF THE INSTANCE (should match the stored data)
PB = "Bayesian"
tweak = 0.5
T = 20000
N = 100
K = 5
nbBreaks = 6

fname="results/$(PB)_tweak_$(tweak)_"

# POLICIES TO INCLUDE
policies=["klUCB","EXP3SOpt","SWKLUCB","DKLUCB","MKLUCB2","CUSUMKLUCB2","GLRKLUCB","GLRKLUCBLoc"]
names =["klUCB","EXP3.S","SW-klUCB","D-klUCB","M-klUCB","CUSUM-klUCB","GLR-klUCB","GLR-klUCB-Local"]


lP = length(policies)

df=DataFrame(Algo= String[],Regret=Float32[],Stdev=Float32[],NbRestarts=Float32[])
clf()

# display results and plot the Bayesian regret 
for imeth in 1:lP
    policy = policies[imeth]
    policyname = names[imeth]
    name = "$(fname)_$(policy)_T_$(T)_N_$(N)_K_$(K)_U_$(nbBreaks)"
    Regret=h5read(name,"Regret")
    Restarts=h5read(name,"Restarts")  
    tsave=floor.(Int,collect(range(1,T,length=500)))
    # print 
    regfinal = mean(Regret[:,end])
    stdevfinal = std(Regret[:,end])
    restarts = mean(Restarts)
    push!(df,("$(policy)",regfinal,stdevfinal,restarts))
    print("Mean final regret for $(policyname) is $(regfinal)\n")
    print("Standard deviation of the final regret for $(policyname) is $(stdevfinal))\n")
	print("Mean number of restarts for $(policyname) is $(restarts)\n\n")
	# plot 
	plot(vec(tsave),vec(mean(Regret,dims=1)),label="$(policyname)")
end

legend()

if (updateCSV == "on")
    CSV.write("$(fname)T_$(T)_N_$(N)_K_$(K)_U_$(nbBreaks).csv",df)
end