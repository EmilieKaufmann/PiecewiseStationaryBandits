# Algorithms for Regret Minimization in a Piecewise Stationary MAB with Bernoulli rewards
# julia 1.0.5

# All algorithms take as an input Table, where Table[i,t] is the rewards of arm i at time t
# All algorithms output ChosenArms,ReceivedRewards,episode (=nb of episodes),ChangePoints (=position of the breakpoints)
# (the last one is always empty for algorithm which do not explicitely perform restarts)

using Distributions

### USEFUL FUNCTIONS

function randmax(vector,rank=1)
   # returns an argmax uniformly at random among all maximizers
   # (integer, not CartesianIndex)
   vector=vec(vector)
   Sorted=sort(vector,rev=true)
   m=Sorted[rank]
   Ind=findall(x->x==m,vector)
   index=Ind[floor(Int,length(Ind)*rand())+1]
   return (index)
end

function sampleDiscrete(probas)
   # return a sample from a distribution on {1,...,K} such that P(X=k)=probas[k]
   cumprobas = cumsum(probas)
   if (cumprobas[end]<0.99)
      print("something is wrong!\n")
   end
   cumprobas[end]=1.
   u = rand()
   k = 1 
   while (u>cumprobas[k])
      k = k+1
   end
   return k
end

function Normalize(weights)
   # cumpute weights/sum(weights), stable for "big" weights
   K = length(weights)
   probas = zeros(Float32,K)
   for k in 1:K
      exptot = 0
      logweight = log(weights[k])
      for j in 1:K
         if (j!=k)
            exptot = exptot + exp(log(weights[j])-logweight)
         end
      end
      probas[k]=1/(1+exptot)
   end
   return probas 
end

function kl(p,q)
   # binary relative entropy
   res=0
   if (p!=q)
      if (p<=0) p = eps() end
      if (p>=1) p = 1-eps() end
      res=(p*log(p/q) + (1-p)*log((1-p)/(1-q))) 
   end
   return res 
end
 
function klUp(p,level)
   # KL upper confidence bound:
   # return uM>p such that d(p,uM)=level 
   lM = p 
   uM = min(min(1,p+sqrt(level/2)),1) 
   for j = 1:16
      qM = (uM+lM)/2
      if kl(p,qM) > level
         uM= qM
      else
         lM=qM
      end
   end
   return uM
end

function klSG(p,q)
   # kl-diverge corresponding to a 1/4-subGaussian approximation
   return 2*(p-q)^2
end

function DrawTable(Problem)
   # generate a Table of rewards from a matrix of mean rewards mu[a,t] 
   (K,T)=size(Problem)
   Table = Int.(rand(K,T).<Problem)
end 

function ComputeCumRegret(Instance,Choices)
   # compute the vector of cumulative regret on a bandit instance based on the successive selections of an algorithm stored in Choices
   K,T = size(Instance)
   regret = 0
   Regret = zeros(Float32,1,T)
   for t in 1:T
      regret+=maximum(Instance[:,t])-Instance[Choices[t],t]
      Regret[t]=regret
   end
   return Regret
end


### ALGORITHMS 

## AdSwitch [Auer et al. 19]

function AdSwitch(Table,C=1,subSample=50,subSample2 = 20,verbose="off")
   # table contains a table of rewards for the K arms up to horizon T
   (K,T)=size(Table)
   # index of the episode 
   episode = 1
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   ChangePoints = []
   t = 0 # current number of sample 
   while (t<T)
      SUMS = zeros(K,T-t,T-t) # array containing the S[a,s,t]
      DRAWS = zeros(K,T-t,T-t) # array containing the N[a,s,t]
      Bad_Gaps = zeros(K) # gaps of the bad arms
      Bad_Means = zeros(K) # means of the bad arms
      tloc = 0 # number of samples in current episode 
      TotalNumber=zeros(Int,K) # total number of selections in episode
      # initialize set of good and bad arms
      Good = collect(1:K)
      Bad = []
      # initialize sampling obligations   
      Obligations = zeros(Int,K)
      newEpisode = false # should we start a new episode?
      # start the episode
      while (!newEpisode)&&(t<T)           
         # form the set of candidate arms to choose from (good arm + sampling obligations)
         Candidate = copy(Good) 
         append!(Candidate,[i for i in 1:K if Obligations[i]>0])
         # draw an arm (least sampled among candidates) 
         I = randmax(-TotalNumber[Candidate])
         rew = Table[I,t+1]
         TotalNumber[I]=+1
         ChosenArms[t+1]=I
         ReceivedRewards[t+1]=rew
         # update everything
         if (Obligations[I]>0)
            # if the arm was sampled due to obligations, recrease the remaining time to sample
            Obligations[I]=Obligations[I]-1
         end
         t=t+1
         tloc=tloc+1
         for i in 1:K
            if (i!=I)&&(tloc>1)
               DRAWS[i,1:(tloc-1),tloc]=DRAWS[i,1:(tloc-1),tloc-1]
               SUMS[i,1:(tloc-1),tloc]=SUMS[i,1:(tloc-1),tloc-1]
            else
               DRAWS[i,tloc,tloc]=1
               SUMS[i,tloc,tloc]=rew
               if (tloc>1)
                  DRAWS[i,1:(tloc-1),tloc]=DRAWS[i,1:(tloc-1),tloc-1].+1
                  SUMS[i,1:(tloc-1),tloc]=SUMS[i,1:(tloc-1),tloc-1].+rew
               end
            end
         end
         # updating the set of good arms 
         for s in 1:tloc 
            cand = [a for a in Good if DRAWS[a,s,tloc]>1]
            candmeans = [SUMS[a,s,tloc]/DRAWS[a,s,tloc] for a in cand]
            if (length(candmeans)>0)
               mumax = maximum(candmeans)
               for i in 1:length(cand)  
                  arm = cand[i]
                  if mumax - candmeans[i] > sqrt(C*log(T)/(DRAWS[arm,s,tloc]-1))
                     if verbose=="on"
                        print("removed arm $(arm) from Good at t=$(t)!\n")
                     end
                     # remove arm from Good 
                     deleteat!(Good, findfirst(isequal(arm), Good))
                     # add arm to bad and store its gap and mean
                     append!(Bad,arm)
                     Bad_Gaps[arm]= mumax - candmeans[i]
                     Bad_Means[arm] = candmeans[i]
                  end
               end
            end
         end
         # perform tests
         if (tloc % subSample ==1)
            # check whether a bad arm has changed 
            check=0
            idchange=0
            s=1
            while (s<tloc)&&(check==0)
               for badarm in Bad 
                  draws=DRAWS[badarm,s,tloc]
                  if (draws>1)&&(abs(SUMS[badarm,s,tloc]/draws-Bad_Means[badarm]) > Bad_Gaps[badarm]/4 + sqrt(2*log(T)/draws))
                     newEpisode = true
                     check +=1
                     idchange=badarm
                  end
               end
               s = s+1
            end
            if (check==0)
               # check whether a good arm has changed
               s = 1
               while (s < tloc)&&(check==0) 
                  for s1 in [j for j in 1:tloc if (j % subSample2 ==1)]
                     for s2 in [j for j in s1:tloc if (j % subSample2 ==1)]
                        for goodarm in Good
                           draws1=DRAWS[goodarm,s1,s2]
                           draws2=DRAWS[goodarm,s,tloc]
                           if (draws1>1)&&(draws2>1)&&(abs(SUMS[goodarm,s1,s2]/draws1 - SUMS[goodarm,s,tloc]/draws2)>sqrt(2*log(T)/draws1)+sqrt(2*log(T)/draws2)) 
                              newEpisode = true
                              check +=1
                              idchange=goodarm
                           end
                        end
                     end
                  end
                  s = s+1
               end
            end
            if (check>0)
               episode+=1
               ChangePoints=append!(ChangePoints,t)
               if verbose=="on"
                  print("detected a change on $(idchange) at t=$(t)\n\n")
               end
            end
         end   
         # possibly add some new sampling obligation 
         for badarm in Bad
            i=1
            while (1/(2^i)>= Bad_Gaps[badarm]/16)
               if (rand() < (1/(2^i))*sqrt(episode/(K*T*log(T))))
                  n =floor(Int,2^(2*i+1)*log(T))
                  if (verbose=="on")
                     print("add a sampling obligation for arm $(badarm) at t=$(t) of length $(n)\n")
                  end
                  # update the sampling obligation
                  Obligations[badarm]=max(Obligations[badarm],n)
               end
               i+=1
            end
         end
      end
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


## Variants of GLR-klUCB 

# GLRklUCB with Global Restart [Besson et al. 20]
# recommended tuning : alpha0 = sqrt(K*log(T)/T) and delta = 1/sqrt(T)

function GLRklUCBGlobal(Table,beta,klChange=kl,alpha0=0.1,delta=0.05,alphatype="auto",subSample=10,subSample2 = 5,verbose="off")
   # beta, klChange : threshold and kl-divergence function used in the GLRT test
   # parameter of the exploration sequence alpha0 (such that the exploration probability in episode k is alpha0*\sqrt(k)), error probability for the test (delta)
   # allow for a constant exploration probability alpha0 for alphatype="constant"
   (K,T)=size(Table)
   # index of the episode 
   episode = 1 # nb of breakpoints detected
   ChangePoints = []
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   t = 0 # current number of sample 
   while (t<T)
      SUMS = [[] for a=1:K] # array containing the cumulated sums of rewards
      tloc = 0 # number of samples in current episode 
      TotalNumber=zeros(Int,K) # total number of selections in current episode
      TotalSum=zeros(Float32,K) # sum of rewards in current episode
      newEpisode = false # should we start a new episode?
      alpha = alpha0*sqrt(episode) # exploration probability for the episode
      if (alphatype == "constant")
         alpha = alpha0
      end
      ExploRange = floor(Int,K/alpha)
      # start the episode
      while (!newEpisode)&&(t<T)
         # default exploratory action
         I = (t % ExploRange)   
         if (I ==0)||(I>K) 
            # compute KL-UCB indices if no exploration
            indices = ones(K)
            for i in 1:K
               if (TotalNumber[i]>0)
                  indices[i]=klUp(TotalSum[i]/TotalNumber[i],log(tloc)/TotalNumber[i])
               end
            end
            I = randmax(indices)
         end
         # get the reward  
         rew = Table[I,t+1]
         ChosenArms[t+1] = I         
         ReceivedRewards[t+1]=rew
         # update everything
         t=t+1
         tloc=tloc+1
         TotalNumber[I]+=1
         TotalSum[I]+=rew
         append!(SUMS[I],TotalSum[I])
         # has the selected arm changed? 
         check = 0
         if (tloc % subSample == 1)
            s = 1 
            nb = TotalNumber[I]
            sums = SUMS[I]
            while (s < nb)&&(check==0)  
               if (s % subSample2 ==1)
                  draw1 = s 
                  draw2 = nb-s
                  mu1 = sums[s]/draw1
                  mu2 = (sums[nb]-sums[s])/draw2 
                  mu = sums[nb]/nb
                  if (draw1*klChange(mu1,mu)+draw2*klChange(mu2,mu))>beta(nb,delta)
                     newEpisode = true
                     check +=1
                  end
               end
               s=s+1
            end
            if (check>0)
               episode+=1
               ChangePoints=append!(ChangePoints,t)
               if verbose=="on"
                  print("detected a change on $(I) at t=$(t)\n\n")
               end
            end
         end   
      end
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


# GLRTklUCB with Local Restarts [Besson et al. 20]
# recommended tuning : alpha0 = sqrt(K*log(T)/T) and delta = 1/sqrt(T)

function GLRklUCBLocal(Table,beta,klChange=kl,alpha0=0.,delta=0.,alphatype="auto",subSample=10,subSample2 = 10,verbose="off")
   # beta, klChange : threshold and kl-divergence function used in the GLRT test
   # parameter of the exploration sequence alpha0 (such that the exploration probability in episode k is alpha0*\sqrt(k)), error probability for the test (delta)
   # allow for a constant exploration probability alpha0 for alphatype="constant"
   (K,T)=size(Table)
   # index of the episode 
   episode = 1 # nb of breakpoints detected
   ChangePoints = []
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   t = 0 # current number of sample 
   SUMS = [[] for a=1:K] # array containing the cumulated sums of rewards
   tloc = zeros(Int,K) # elapsed time since last restart for each arm 
   TotalNumber=zeros(Int,K) # total number of selections since last restart
   TotalSum=zeros(Float32,K) # sum of rewards since last restart
   while (t<T)
      newEpisode = false # should we do a restart on the last arm selects?
      alpha = alpha0*sqrt(episode) # exploration probability for the episode
      if (alphatype == "constant")
         alpha = alpha0
      end
      ExploRange = floor(Int,K/alpha)
      # start the episode
      while (!newEpisode)&&(t<T)
         # default exploratory action
         I = (t % ExploRange)   
         if (I ==0)||(I>K) 
            # compute KL-UCB indices and choose the argmax if no exploration
            indices = ones(K)
            for i in 1:K
               if (TotalNumber[i]>0)
                  indices[i]=klUp(TotalSum[i]/TotalNumber[i],log(tloc[i])/TotalNumber[i])
               end
            end
            I = randmax(indices)
         end
         # get the reward  
         rew = Table[I,t+1]
         ChosenArms[t+1] = I         
         ReceivedRewards[t+1]=rew
         # update everything
         t=t+1
         tloc=tloc.+1 # update ALL the local times
         TotalNumber[I]+=1
         TotalSum[I]+=rew
         append!(SUMS[I],TotalSum[I])
         # has the selected arm changed? 
         check = 0
         if (tloc[I] % subSample == 1)
            s = 1 
            nb = TotalNumber[I]
            sums = SUMS[I]
            while (s < nb)&&(check==0)  
               if (s % subSample2 ==1)
                  draw1 = s 
                  draw2 = nb-s
                  mu1 = sums[s]/draw1
                  mu2 = (sums[nb]-sums[s])/draw2 
                  mu = sums[nb]/nb
                  if (draw1*klChange(mu1,mu)+draw2*klChange(mu2,mu))>beta(nb,delta)
                     newEpisode = true
                     check +=1
                  end
               end
               s=s+1
            end
            if (check>0)
               episode+=1
               ChangePoints=append!(ChangePoints,t)
               if verbose=="on"
                  print("detected a change on $(I) at t=$(t)\n\n")
               end
               # reset the history of arm I only 
               tloc[I]=0
               SUMS[I]=[]
               TotalNumber[I]=0 
               TotalSum[I]=0 
            end
         end   
      end
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


## Other approaches based on CPD 

# M-(kl)UCB adaptation from [Cao et al. 20]
# recommended tuning: w of order log(2KT^2)/(size of the smallest change)^2 [or other domain-specific tuning]
# b = sqrt(w*log(2*K*T^2)/2)
# alpha = sqrt((nbreaks*K*(2*b+3*\sqrt(w)))/(2*T)) 

function MklUCB(Table,alpha=0.1,w=150,b=50.,verbose="off")
   # exploration parameter alpha (=gamma in the paper), windows w, threshold b
   (K,T)=size(Table)
   # index of the episode 
   episode = 1 # nb of breakpoints detected
   ChangePoints = []
   ExploRange = floor(Int,K/alpha)
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   t = 0 # current number of sample 
   while (t<T)
      SUMS = [[] for a=1:K] # array containing the cumulated sums of rewards
      tloc = 0 # number of samples in current episode 
      TotalNumber=zeros(Int,K) # total number of selections since last restart
      TotalSum=zeros(Float32,K) # sum of rewards since last restart
      newEpisode = false # should we start a new episode?
      # start the episode
      while (!newEpisode)&&(t<T)
         # default exploratory action
         I = (t % ExploRange)   
         if (I ==0)||(I>K) 
            # compute KL-UCB indices if no exploration
            indices = ones(K)
            for i in 1:K
               if (TotalNumber[i]>0)
                  indices[i]=klUp(TotalSum[i]/TotalNumber[i],log(tloc)/TotalNumber[i])
               end
            end
            I = randmax(indices)
         end
         # get the reward  
         rew = Table[I,t+1]
         ChosenArms[t+1] = I         
         ReceivedRewards[t+1]=rew
         # update everything
         t=t+1
         tloc=tloc+1
         TotalNumber[I]+=1
         TotalSum[I]+=rew
         append!(SUMS[I],TotalSum[I])
         # has the selected arm changed? 
         nb = TotalNumber[I]
         if (nb >= w)
            sums = SUMS[I]   
            mid = sums[nb-floor(Int,w/2)]
            FirstHalf = mid - sums[nb-w+1]
            SecondHalf = sums[nb]-mid
            if (abs(FirstHalf-SecondHalf)> b)
               newEpisode = true
               episode+=1
               ChangePoints=append!(ChangePoints,t)
               if verbose=="on"
                  print("detected a change on $(I) at t=$(t)\n\n")
               end
            end
         end   
      end
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


# CUSUM-(kl)UCB adaptation from [Liu et al 2018] 
# recommended tuning h = log(T/nbreaks) and alpha=sqrt(nbreaks*log(T/nbreaks)/T)
# M and epsilon require some prior knowledge of the problem 

function CUSUMklUCB(Table,alpha=0.05,M=150,epsilon=0.1,h=50,verbose="off")
   # exploration parameter alpha, intial nb of sample M, minimum change epsilon, threshold h 
   (K,T)=size(Table)
   # index of the episode 
   episode = 1 # nb of breakpoints detected
   ChangePoints = [] 
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   t = 0 # current number of sample 
   TotalNumber=zeros(Int,K) # total number of selections since last restart
   TotalSum=zeros(Float32,K) # sum of rewards since last restart
   # random walks associated to each arm used for stopping 
   GPlus = zeros(Float32,K) # first random walk associated to each arm (used in CUSUM test)
   GMinus = zeros(Float32,K) # second random walk associated to each arm
   InitialAverage = zeros(Float32,K) # average of the first M observations from each arm
   newEpisode = false # should we start a new episode?
   while (t<T)
      newEpisode = false # should we start a new episode?
      # start the episode
      while (!newEpisode)&&(t<T)
         I = 0
         if (rand()<alpha)
            # explore uniformly at random
            I = rand(1:K)
         else 
            # compute KL-UCB indices and choose the argmax if no exploration
            indices = ones(K)
            nvalid = sum(TotalNumber)
            for i in 1:K
               if (TotalNumber[i]>0)
                  indices[i]=klUp(TotalSum[i]/TotalNumber[i],log(nvalid)/TotalNumber[i])
               end
            end
            I = randmax(indices)
         end
         # get the reward  
         rew = Table[I,t+1]
         ChosenArms[t+1] = I         
         ReceivedRewards[t+1]=rew
         # update everything
         t=t+1
         TotalNumber[I]+=1
         TotalSum[I]+=rew
         if (TotalNumber[I] == M)
            InitialAverage[I] = TotalSum[I]/TotalNumber[I]         
         elseif (TotalNumber[I]>M)
            GPlus[I]=max(0,GPlus[I]+rew-InitialAverage[I]-epsilon)
            GMinus[I]=max(0,GMinus[I]+InitialAverage[I]-rew-epsilon)
         end
         # has the selected arm changed? 
         if (GMinus[I]>h)||(GPlus[I]>h)
            newEpisode = true
            episode+=1
            ChangePoints=append!(ChangePoints,t)
            if verbose=="on"
               print("detected a change on $(I) at t=$(t)\n\n")
            end
            # reset the history of arm I only 
            TotalNumber[I]=0 
            TotalSum[I]=0 
            InitialAverage[I]=0
            GPlus[I]=0
            GMinus[I]=0
         end   
      end
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


##  EXP3.S [Auer et al. 02]
# recommended tuning alpha=1/T and gamma = min(1,sqrt(K*(nbreak*log(K*T)+exp(1))/((exp(1)-1)*T))) 
# (automatic tuning if nbreak is given)

function EXP3S(Table,nbreak=0,gamma=0.1,alpha=0.01)
   # table contains a table of rewards for the K arms up to horizon T
   (K,T)=size(Table)
   episode=1
   ChangePoints = []
   if (nbreak>0)
      # optimized gamma as a function of the nb of breakpoints 
      gamma = min(1,sqrt(K*(nbreak*log(K*T)+exp(1))/((exp(1)-1)*T)))
      alpha = 1/T
   end
   Weights=(1/K)*ones(Float32,K) # vector of weights / probability to sample
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   for t in 1:T
      Probas = (1-gamma)*Weights .+ gamma/K
      I = sampleDiscrete(Probas)
      # get the reward  
      rew = Table[I,t]
      ChosenArms[t]=I 
      ReceivedRewards[t]=rew
      # update the weights
      normrew = rew/Probas[I]
      bonus = (exp(1)*alpha/K)
      for k in 1:K
         if k!=I
            Weights[k]=Weights[k]+bonus
         else
            Weights[k]=(exp(gamma*normrew/K))*Weights[k]+bonus
         end
      end
      # normalization step: storing the normalized weights and working with them leads to the same algorithm
      Weights = Normalize(Weights)
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end



## Passively adaptive algorithms 

# Discounted-(kl)UCB adaptation of [Kocsis et al. 06,Garivier and Moulines 08] 
# recommended tuning: 1- 0.25*sqrt(nbBreaks/T)
# (automatic tuning if nbreak is given)

function DklUCB(Table,nbreak=0,gamma=0.95)
   # table contains a table of rewards for the K arms up to horizon T
   (K,T)=size(Table)
   episode=1
   ChangePoints = []
   if (nbreak>0)
      # optimized gamma as a function of the nb of breakpoints 
      gamma = 1- 0.25*sqrt(nbBreaks/T)
   end
   DiscNumber=zeros(Float32,K) # discounted number of selections 
   DiscSum=zeros(Float32,K) # discounted sum of rewards
   Time =0 # discounted total number of pulls 
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   for t in 1:T
      # compute KL-UCB indices 
      indices = ones(K)
      for i in 1:K
         if (DiscNumber[i]>0)
            indices[i]=klUp(DiscSum[i]/DiscNumber[i],log(Time)/DiscNumber[i])
         end
      end
      I = randmax(indices)
      # get the reward  
      rew = Table[I,t]
      ChosenArms[t]=I 
      ReceivedRewards[t]=rew
      # update everything
      DiscNumber = gamma*DiscNumber
      DiscNumber[I]+=1
      DiscSum = gamma*DiscSum
      DiscSum[I]+=rew
      Time = 1+gamma*Time
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


# SW-(kl)UCB adaptation of [Garivier and Moulines 08] 
# recommended tuning:  tau = 2*sqrt(T*log(T)/nbreak)
# (automatic tuning if nbreak is given)

function SWklUCB(Table,nbreak=0,tau=450)
   # table contains a table of rewards for the K arms up to horizon T
   (K,T)=size(Table)
   episode=1
   ChangePoints = []
   if (nbreak>0)
      # optimized tau as a function of the nb of breakpoints 
      tau = floor(Int,2*sqrt(T*log(T)/nbreak))
   end
   TabNumber=zeros(Int,T,K) # N[t,k] : nb of selections of arm k in first t rounds 
   TabSum=zeros(Float32,T,K) # S[t,k] : sum of rewards from arm k in first t rounds
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   for t in 1:T
      # compute KL-UCB indices 
      indices = ones(K)
      NumberLoc = TabNumber[max(t-1,1),:]
      SumLoc = TabSum[max(t-1,1),:] 
      if (t>tau)
         NumberLoc = TabNumber[t-1,:]-TabNumber[t-tau,:]
         SumLoc = TabSum[t-1,:] - TabSum[t-tau,:]
      end
      for i in 1:K
         if (NumberLoc[i]>0)
            indices[i]=klUp(SumLoc[i]/NumberLoc[i],log(min(t,tau))/NumberLoc[i])
         end
      end
      I = randmax(indices)
      # get the reward  
      rew = Table[I,t]
      ChosenArms[t]=I 
      ReceivedRewards[t]=rew
      # update everything
      TabNumber[t,:] = TabNumber[max(t-1,1),:]
      TabNumber[t,I]=TabNumber[t,I]+1
      TabSum[t,:] = TabSum[max(t-1,1),:]
      TabSum[t,I]=TabSum[t,I]+rew
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end



# D-TS, investigated by [Raj and Kalyani 2017]
# recommended tuning is gamma=0.75, which is not so robust

function DTS(Table,gamma=0.75,a=1,b=1)
   # discount gamma, Beta(a,b) prior 
   (K,T)=size(Table)
   episode=1
   ChangePoints = []
   DiscNumber=zeros(Float32,K) # discounted number of selections 
   DiscSum=zeros(Float32,K) # discounted sum of rewards
   Time =0 # discounted total number of pulls 
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   for t in 1:T
      # compute KL-UCB indices 
      indices = ones(K)
      for i in 1:K
         if (DiscNumber[i]>0)
            indices[i]=rand(Beta(a+DiscSum[i], b+DiscNumber[i]-DiscSum[i]), 1)[1]
         end
      end
      I = randmax(indices)
      # get the reward  
      rew = Table[I,t]
      ChosenArms[t]=I 
      ReceivedRewards[t]=rew
      # update everything
      DiscNumber = gamma*DiscNumber
      DiscNumber[I]+=1
      DiscSum = gamma*DiscSum
      DiscSum[I]+=rew
      Time = 1+gamma*Time
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end


## Variants of kl-UCB [Cappé et al 2013]

function klUCB(Table)
   # table contains a table of rewards for the K arms up to horizon T
   (K,T)=size(Table)
   episode=1
   ChangePoints = []
   # index of the episode 
   TotalNumber=zeros(Int,K) # total number of selections 
   TotalSum=zeros(Float32,K) # sum of rewards 
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   for t in 1:T
      # compute KL-UCB indices 
      indices = ones(K)
      for i in 1:K
         if (TotalNumber[i]>0)
            indices[i]=klUp(TotalSum[i]/TotalNumber[i],log(t)/TotalNumber[i])
         end
      end
      I = randmax(indices)
      # get the reward  
      rew = Table[I,t]
      ChosenArms[t]=I 
      ReceivedRewards[t]=rew
      # update everything
      TotalNumber[I]+=1
      TotalSum[I]+=rew
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end

function OracleklUCB(Table,Breaks=[])
   # requires a vector Breaks giving the positions of the breakpoints
   (K,T)=size(Table)
   ChosenArms = zeros(Int,T)
   ReceivedRewards = zeros(T)
   # episodes and change-points are predetermined
   episode = length(Breaks)+1 
   ChangePoints = copy(Breaks)
   # important times 
   Times = copy(Breaks)
   append!(Times,T)
   pushfirst!(Times,1)
   for k in 1:episode
      TotalNumber=zeros(Int,K) # total number of selections during episode
      TotalSum=zeros(Float32,K) # sum of rewards 
      for t in Times[k]:Times[k+1]
         # compute KL-UCB indices 
         indices = ones(K)
         tloc = t -  Times[k] + 1
         for i in 1:K
            if (TotalNumber[i]>0)
               indices[i]=klUp(TotalSum[i]/TotalNumber[i],log(tloc)/TotalNumber[i])
            end
         end
         I = randmax(indices)
         # get the reward  
         rew = Table[I,t]
         ChosenArms[t]=I 
         ReceivedRewards[t]=rew
         # update everything
         TotalNumber[I]+=1
         TotalSum[I]+=rew
      end
   end
   return ChosenArms,ReceivedRewards,episode,ChangePoints
end

