# Visualize results on a particular problem that are saved in the results folder (names should match)

using Statistics
using PyPlot
using HDF5
using DataFrames
using CSV

include("Benchmarks.jl")


# OPTIONAL: write results in a csv file
updateCSV = "on"
 
# NAME OF THE INSTANCE (should match the stored data in the results/ folder)
PB = "PB1"
tweak = 1
T = 20000
N = 100

# INCLUDE ADSWITCH? 
withAdSwitch = "off"
NAd = 50 # number of repetitions used for AdSwitch (can be different from N)


fname="results/$(PB)_tweak_$(tweak)_"

# POLICIES TO INCLUDE (check that you have data for them)
policies=["Oracle","klUCB","EXP3SOpt","SWKLUCB","DKLUCB","MKLUCB","MKLUCB2","CUSUMKLUCB","CUSUMKLUCB2","GLRKLUCB","GLRKLUCBConstant","GLRKLUCBLoc","GLRKLUCBLocConstant"]
names=["Oracle","klUCB","EXP3SOpt","SWKLUCB","DKLUCB","MKLUCB","MKLUCB2","CUSUMKLUCB","CUSUMKLUCB2","GLRKLUCB","GLRKLUCBConstant","GLRKLUCBLoc","GLRKLUCBLocConstant"]

# sub-sample of policies for the regret curves
#policies=["Oracle","klUCB","EXP3SOpt","SWKLUCB","DKLUCB","MKLUCB2","CUSUMKLUCB2","GLRKLUCB","GLRKLUCBLoc"]
#names =["Oracle","klUCB","EXP3.S","SW-klUCB","D-klUCB","M-klUCB","CUSUM-klUCB","GLR-klUCB Global","GLR-klUCB Local"]


lP = length(policies)

df=DataFrame(Algo= String[],Regret=Float32[],Stdev=Float32[],NbRestarts=Float32[])
clf()

# plot the problem 
name = "$(fname)_$(policies[1])_T_$(T)_N_$(N)"
Problem = h5read(name,"Problem")
clf()
PlotProblem(Problem)
figure()

# display results and plot the regret 
for imeth in 1:lP
    policy = policies[imeth]
    policyname = names[imeth]
    name = "$(fname)_$(policy)_T_$(T)_N_$(N)"
    Regret=h5read(name,"Regret")
    Restarts=h5read(name,"Restarts")  
    tsave=floor.(Int,collect(range(1,T,length=500)))
    # print stuff
    regfinal = mean(Regret[:,end])
    stdevfinal = std(Regret[:,end])
    restarts = mean(Restarts)
    push!(df,("$(policy)",regfinal,stdevfinal,restarts))
    print("Mean final regret for $(policyname) is $(regfinal)\n")
    print("Standard deviation of the final regret for $(policyname) is $(stdevfinal))\n")
	print("Mean number of restarts for $(policyname) is $(restarts)\n\n")
	# plot regret
	plot(vec(tsave),vec(mean(Regret,dims=1)),label="$(policyname)")
end

if withAdSwitch == "on"
    nameAd = "$(fname)_AdSwitch_T_$(T)_N_$(NAd)"
    Regret = h5read(nameAd,"Regret")
    Restarts=h5read(nameAd,"Restarts")
    regfinal = mean(Regret[:,end])
    stdevfinal = std(Regret[:,end])
    restarts = mean(Restarts)
    push!(df,("AdSwitch",regfinal,stdevfinal,restarts))
    tsave=floor.(Int,collect(range(1,T,length=500)))
    print("Mean final regret for AdSwitch is $(regfinal)\n")
    print("Standard deviation of the final regret for AdSwitch is $(stdevfinal))\n")
	print("Mean number of restarts for AdSwitch is $(restarts)\n\n")
    plot(vec(tsave),vec(mean(Regret,dims=1)),label="AdSwitch")
end

legend()

if (updateCSV == "on")
    CSV.write("$(fname)T_$(T)_N_$(N).csv",df)
end