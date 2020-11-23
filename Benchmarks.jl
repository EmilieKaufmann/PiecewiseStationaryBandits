# Useful functions to create piecewise stationary bandit instances, stored in matrix of means:  Problem[a,t] is the mean of arm a at time t

using PyPlot

function NoChange(T,mu)
    # a stationary bandit problem with means mu 
    K = length(mu)
    Problem = zeros(K,T)
    for k in 1:K
        Problem[k,:]=mu[k]*ones(T)
    end
    return Problem,zeros
end

function ProblemUnif(T,MeansMatrix)
    # generate means rewards with uniformly spaced breakpoints 
    # MeansMatrix[a,k] gives the value of \mu_a^{(k+1)}
    (K,Episodes)=size(MeansMatrix)
    nbBreaks = Episodes - 1
    Problem = zeros(K,T)
    # size of each episode
    part = round(Int,T/(nbBreaks+1))
    # filling the matrix 
    for c in 0:nbBreaks
	    for arm in 1:K
            Problem[arm,(1+c*part):(c+1)*part]=Problem[arm,(1+c*part):(c+1)*part].+MeansMatrix[arm,c+1]
        end
    end
    if (Episodes*part < T)
        for arm in 1:K
            Problem[arm,(Episodes*part+1):T]=Problem[arm,(Episodes*part+1):T].+MeansMatrix[arm,Episodes]
        end
    end
    BreakPoints = [i*part for i in 1:nbBreaks]
    return Problem,BreakPoints
end

function ProblemGeneral(MeansMatrix,Landmarks)
    # between Landmarks[k] and Landmarks[k+1] the mean of arm a is MeansMatrix[a,k]
    # one must have Landmarks[1]=1 and Landmarks[L]=T
    L = length(Landmarks)
    (K,l) = size(MeansMatrix)
    if L>(l+1)
        return "wrong dimensions"
    else
        T= Landmarks[L]
        Problem = zeros(Float32,K,T)
        for k in 1:(L-1)
            for arm in 1:K
                Problem[arm,Landmarks[k]:Landmarks[k+1]]=MeansMatrix[arm,k]*ones(Float32,Landmarks[k+1]-Landmarks[k]+1)
            end
        end
        return Problem,Landmarks[2:(L-1)]
    end
end

function RandomProblem(K,T,nbreaks,p,Deltamin = 0.05,Deltamax = 0.3,Spacemin=500)
    # generates a random instance with K arms up to horizon T with at most nbreaks breakpoint
    # p: probability that a change occurs for each arm in each breakpoint
    # Deltamin - Deltamax : minimal - maximal magnitude of a change-point
    # Spacemin : minimal distance between two breakpoints 
    cont = true
    # choose the breakpoint (resample until they match the Spacemin condition)
    Landmarks = zeros(nbreaks+2)
    while (cont)
        cont = false
        Landmarks = sort(floor.(Int,T*rand(nbreaks+2))) 
        Landmarks[1]=1
        Landmarks[nbreaks+2]=T
        for i in 1:(nbreaks+1)
            if ((Landmarks[i+1]-Landmarks[i])<Spacemin)
                cont = true
            end
        end
    end
    # matrix of means 
    MeansMatrix = zeros(K,nbreaks+1)
    # random initial means
    for a in 1:K
        MeansMatrix[a,1]=rand()
    end
    # in each breakpoint, there is a probability p that and arm keeps its previous value 
    for k in 2:(nbreaks+1)
        for a in 1:K
            prevmean = MeansMatrix[a,k-1]
            if (rand()>p)
                MeansMatrix[a,k]=prevmean
            else
                gap = Deltamin + (Deltamax - Deltamin)*rand()
                if (rand()<0.5)
                    gap = -gap
                end
                if ((prevmean + gap)>1)||((prevmean + gap)<0)
                    MeansMatrix[a,k] = prevmean-gap
                else
                    MeansMatrix[a,k] = prevmean + gap
                end
            end
        end
    end
    return ProblemGeneral(MeansMatrix,Landmarks)
end  

function ProblemToMatrix(Problem,nbreaks)
    # return the matrix of means and landmarks that allow to generate the problem
    (K,T)=size(Problem)
    Landmarks = zeros(Int,nbreaks+2)
    MeansMatrix = zeros(K,nbreaks+1)
    ndetect = 0
    MeansMatrix[:,1]=MeansMatrix[:,1].+Problem[:,1]
    Landmarks[1]=1
    for t = 2:T
        if (Problem[:,t]!=Problem[:,t-1])
            ndetect = ndetect+1
            Landmarks[ndetect+1]=t
            MeansMatrix[:,ndetect+1]=MeansMatrix[:,ndetect+1].+Problem[:,t]
        end
    end
    Landmarks[ndetect+2]=T
    return MeansMatrix[:,1:(ndetect+1)],Landmarks[1:(ndetect+2)]
end

function PlotProblem(Pb)
    # visualize the evolution of means in a bandit instance
    (K,T)=size(Pb)
    for k in 1:K
        plot(1:T,Pb[k,:],label="arm $(k)")
    end
    legend()
    return 
end


# matrix of means corresponding to Problem1 in the paper
MeansMatrix1 = reshape([0.9,0.5,0.3,0.9,0.2,0.3,0.1,0.2,0.3,0.1,0.7,0.3,0.1,0.7,0.5],3,5)
# matrix of means corresponding to Problem 2 in the paper
MeansMatrix2 = reshape([0.9,0.5,0.4,0.7,0.4,0.5,0.5,0.3,0.6,0.3,0.2,0.7,0.1,0.1,0.8],3,5)

# a few other saved Benchmarks (before trying any expes on them)
# WorseBecomeGood : changes not evenly spaced, worse arm eventually becomes good (PB3 in the paper)
# OneOptChange: only one meaningful change (PB4 in the paper)