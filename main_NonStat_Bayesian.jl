# run bandit algorithms on piecewise stationary bandit instance (and possibly save results)
# the choice of parameters given in this example correspond to the Bayesian experiment (average of many random problem instances) studied in Table 1 of [Besson et al. 20] with alpha0=0.5

using HDF5
using Statistics
using PyPlot

include("Algos.jl")
include("Benchmarks.jl")

# SAVING RESULTS
Save = "on" # should we save results ?

# TYPE OF EXPERIMENT
Type = "Bayesian" # Frequentist or Bayesian experiment?
# NUMBER OF REPETITIONS
N = 100 
# TIME HORIZON  
T = 20000 

# NAME YOUR INSTANCE 
# default choices : PB1, PB2, PB3, PB4 [see Besson et al. 2020], Random 
# or use an other name for Bayesian experiments or for a custom problem
PB = "Bayesian"

# PARAMETERS OF THE RANDOM INSTANCES SAMPLER
K = 5 
nbBreaks = 6 
p=0.5 # probability that a breakpoint is a CP for each arm 
Deltamin = 0.05 # minimal magnitude of a change
Deltamax = 0.3 # maximal magnitude of a change
Spacemin=1000 # minimal space between two breakpoints


# CUSTOM NON-STATIONARY PROBLEM (only necessary if you didn't choose one of the default names) 
# (see different ways to specify them in Benchmarks.jl)
Problem,BreakPoints = ProblemUnif(T,MeansMatrix1)


if (PB == "PB1")
	# Problem 1 in [Besson et al. 2020]
	Problem,BreakPoints = ProblemUnif(T,MeansMatrix1)
elseif (PB == "PB2")
	# Problem 2 in [Besson et al. 2020]
	Problem, BreakPoints = ProblemUnif(T,MeansMatrix2)
elseif (PB =="PB3")
	# Problem 3 in [Besson et al. 2020]
	Problem = h5read("benchmarks/WorseBecomeGood","Problem")
	BreakPoints = h5read("benchmarks/WorseBecomeGood","BreakPoints")
elseif (PB =="PB4")
	# Problem 4 in [Besson et al. 2020]
	Problem = h5read("benchmarks/OneOptChange","Problem")
	BreakPoints = h5read("benchmarks/OneOptChange","BreakPoints")
elseif (PB[1:6] == "Random")
	# select a problem at random from the above sampler
	Problem, BreakPoints = RandomProblem(K,T,nbBreaks,p,Deltamin,Deltamax,Spacemin)
end	

# display the arms means 
if Type=="Frequentist"
	nbBreaks=length(BreakPoints)
	(K,T)=size(Problem)
	clf()
	PlotProblem(Problem)
end




## parameters for the CPD-based algorithms 

# TWEAK 
tweak = 0.5 # multiplicative constant in front of the exploration parameter alpha for all CDP-based algorithms

alpha0 = tweak*sqrt(K*log(T)/T)
alpha_opt = alpha0*sqrt(nbBreaks) # for algorithms using the knwoledge of nbBreaks
delta_GLR = 1/sqrt(T) # GLR-klUCB
M_CUSUM = 200 # CUMSUM (two variants tried in the paper)
epsilon_CUSUM = 0.1
M_CUSUM2 = 400 
epsilon_CUSUM2 = Deltamin
h_CUSUM = log(T/nbBreaks)
w_MUCB = 200 # MUCB (two variants tried in the paper)
b_MUCB = sqrt(w_MUCB*log(2*K*T^2)/2)
w_MUCB2 = 800 
b_MUCB2 = sqrt(w_MUCB2*log(2*K*T^2)/2)

# print a summary of the tuning
print("Exploration parameters are alpha0=$(alpha0), alpha_opt=$(alpha_opt)\n")
print("Parameters for GLR-klUCB: delta = $(delta_GLR)\n")
print("Parameters for CUSUM-klUCB: M = $(M_CUSUM), epsilon = $(epsilon_CUSUM), h = $(h_CUSUM)\n")
print("Parameters for M-klUCB: windows w = $(w_MUCB), threshold b = $(b_MUCB)\n")
print("Parameters for CUSUM-klUCB 2: M = $(M_CUSUM2), epsilon = $(epsilon_CUSUM2), h = $(h_CUSUM)\n")
print("Parameters for M-klUCB 2: windows w = $(w_MUCB2), threshold b = $(b_MUCB2)\n")

# name of the expes 
series = "$(PB)_tweak_$(tweak)_"

## defining algorithms

# GLR-klUCB
beta(n,delta)=log(n^(3/2)/delta)
# Bernoulli GLR
GLRKLUCB(Table) = GLRklUCBGlobal(Table,beta,kl,alpha0,delta_GLR)
GLRKLUCBLoc(Table) = GLRklUCBLocal(Table,beta,kl,alpha0,delta_GLR)
# Gaussian GLR
GLRKLUCBGauss(Table) = GLRklUCBGlobal(Table,beta,klSG,alpha0,delta_GLR)
GLRKLUCBLocGauss(Table) = GLRklUCBLocal(Table,beta,klSG,alpha0,delta_GLR)
# Bernoulli GLR with constant, optimized, exploration 
GLRKLUCBConstant(Table) = GLRklUCBGlobal(Table,beta,kl,alpha_opt,delta_GLR,"constant")
GLRKLUCBLocConstant(Table) = GLRklUCBLocal(Table,beta,kl,alpha_opt,delta_GLR,"constant")

# Oracle tuned as a function of the positions of breakpoints
Oracle(Table) = OracleklUCB(Table,BreakPoints)

# Other algorithms
SWKLUCB(Table) = SWklUCB(Table,nbBreaks)
DKLUCB(Table) = DklUCB(Table,nbBreaks)
EXP3SOpt(Table) = EXP3S(Table,nbBreaks)
MKLUCB(Table) = MklUCB(Table,alpha_opt,w_MUCB,b_MUCB)
MKLUCB2(Table) = MklUCB(Table,alpha_opt,w_MUCB2,b_MUCB2)
CUSUMKLUCB(Table) = CUSUMklUCB(Table,alpha_opt,M_CUSUM,epsilon_CUSUM,h_CUSUM)
CUSUMKLUCB2(Table) = CUSUMklUCB(Table,alpha_opt,M_CUSUM2,epsilon_CUSUM2,h_CUSUM)

print("Parameters for SW-klUCB: windows size tau = $(floor(Int,2*sqrt(T*log(T)/nbBreaks)))\n")
print("Parameters for D-klUCB: discount gamma = $(1- 0.25*sqrt(nbBreaks/T))\n\n")

## SELECT THE POLICIES TO RUN 
#policies=[Oracle,klUCB,EXP3SOpt,SWKLUCB,DKLUCB,MKLUCB,MKLUCB2,CUSUMKLUCB,CUSUMKLUCB2,GLRKLUCB,GLRKLUCBConstant,GLRKLUCBLoc,GLRKLUCBLocConstant]
#policies=[Oracle,klUCB,EXP3SOpt,SWKLUCB,DKLUCB,MKLUCB2,CUSUMKLUCB2,GLRKLUCB,GLRKLUCBLoc]
policies=[klUCB,EXP3SOpt,SWKLUCB,DKLUCB,MKLUCB2,CUSUMKLUCB2,GLRKLUCB,GLRKLUCBLoc]


function MultiExpes(Problem,N,policies,series="test",display="on",saveRes="off")
	# run N repetitions of the chosen policies on the bandit with means given in Problem 
	# optional: regret plot / saving the data
	(K,T)=size(Problem)
	lP = length(policies)
	# sub-sample the time 
	tsave=floor.(Int,collect(range(1,T,length=500)))'
	ts = length(tsave)
	# draw N Tables of reward ahead of time
	Tables = [DrawTable(Problem) for n in 1:N]
	if display=="on"
		figure()
	end
	for imeth in 1:lP
		starttime = time()
		policy = policies[imeth]
		name = "results/$(series)_$(policy)_T_$(T)_N_$(N)"	
		# storing regret and restarts
		Regret = zeros(N,ts)
		Restarts = zeros(N)	
		for n in 1:N
			ChosenArms,ReceivedRewards,episode,ChangePoints= policy(Tables[n])
			reg = ComputeCumRegret(Problem,ChosenArms)
			Regret[n,:]=vec(reg[tsave])
			Restarts[n]=length(ChangePoints)
		end
		print("Results for $(policy)\n")
		print("Mean final regret is $(mean(Regret[:,end]))\n")
		print("Standard deviation of the final regret is $(std(Regret[:,end]))\n")
		print("Mean number of restarts is $(mean(Restarts))\n")
		print("Elapsed time is $(time()-starttime)\n\n")
		if (display == "on")
			# plot regret
			plot(tsave,mean(Regret,dims=1)',label="$(policy)")
		end
		if (saveRes == "on")
			# save date
			h5write(name,"Problem",Problem)
        	h5write(name,"Regret",Regret)
			h5write(name,"Restarts",Restarts)
		end 
	end
	if (display == "on")
		legend()
	end
	return 
end


function BayesianExpes(N,policies,K,T,nbBreaks,p,Deltamin,Deltamax,Spacemin,series="test",saveres="on")
	# run each policy on N random instances drawn from the sampler 
	# optional : save the results
	lP = length(policies)
	# sub-sample the time 
	tsave=floor.(Int,collect(range(1,T,length=500)))'
	ts = length(tsave)
	# draw the N Tables for N different random instances ahead of time 
	Problems = [RandomProblem(K,T,nbBreaks,p,Deltamin,Deltamax,Spacemin)[1] for n in 1:N]
	Tables = [DrawTable(Pb) for Pb in Problems]
	for imeth in 1:lP
		starttime = time()
		policy = policies[imeth]
		name = "results/$(series)_$(policy)_T_$(T)_N_$(N)_K_$(K)_U_$(nbBreaks)"	
		Regret=zeros(N,ts)
		Restarts=zeros(N)
		for n in 1:N
			Problem = Problems[n]
			ChosenArms,ReceivedRewards,episode,cpd= policy(Tables[n])
			reg = ComputeCumRegret(Problem,ChosenArms)
			Regret[n,:] = reg[tsave]
			Restarts[n] = length(cpd)
		end
		print("Results for $(policy)\n")
		print("Mean final regret is $(mean(Regret[:,end]))\n")
		print("Mean number of restarts is $(mean(Restarts))\n")
		print("Elapsed time is $(time()-starttime)\n\n")
		# save data
		if (saveres=="on")
	    	h5write(name,"Regret",Regret)
			h5write(name,"Restarts",Restarts) 
			h5write(name,"Deltamin",Deltamin) 
			h5write(name,"Deltamax",Deltamax)
			h5write(name,"Spacemin",Spacemin)
			h5write(name,"probaCP",p)
		end			
	end
	return 
end

start=time()


if (Type=="Frequentist")
	if Save == "on"
		# saving the data (no visualization)
		MultiExpes(Problem,N,policies,series,"off","on")
	else
		# run expes and display a regret plot
		MultiExpes(Problem,N,policies,series,"on","off")
	end
else 
	if Save == "on"
		BayesianExpes(N,policies,K,T,nbBreaks,p,Deltamin,Deltamax,Spacemin,series)
	else
		BayesianExpes(N,policies,K,T,nbBreaks,p,Deltamin,Deltamax,Spacemin,series,"off")
	end
end


print("total elapsed time is $(time()-start)")