#!/usr/bin/python
############################################################################################################
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import re
import sys

np.set_printoptions(precision=4)
#====================================================      
#====================================================
#====================================================
# GENETIC ALGORITHM
#====================================================
#====================================================
#====================================================
class genetic_algorithm(object):

    # =======================================================
    def __init__(self,fct,xlo,xup,ylo,yup,glo,gup,pop_init=None,**kwargs):
        # -----------------------------
        self.fct=fct
        self.xlo,self.xup=np.asarray(xlo),np.asarray(xup)
        self.ylo,self.yup=np.asarray(ylo),np.asarray(yup)
        self.glo,self.gup=np.asarray(glo),np.asarray(gup)
        # -----------------------------
        # Optimizer Settings
        # -----------------------------
        self.settingsdict={}
        self.settingsdict["strategy"]="genetic_algorithm"
        self.settingsdict["mutate"]=0.05
        self.settingsdict["dbbest_paretorank"]=2
        self.settingsdict["fitness_manager"]=[1.0,1.0,1.0,1.0]
        self.settingsdict["no_childs"]=2
        self.settingsdict["randomcrossover"]=False
        self.settingsdict["itmax"]=1000
        self.settingsdict["npop"]=20
        self.settingsdict["2D_testfunction"]=False
        self.settingsdict["2D_mo_testfunction"]=False
        self.settingsdict["niching"]=False
        self.settingsdict["fitness_evalutation_mode"]="serial" # "parallel" / "serial"
        self.settingsdict["penalty_function"]="absolute" # "absolute" / "quadratic" / "ln"

        for key in kwargs:
            self.settingsdict[key]=kwargs[key]
        #--------------------------------
        # Initialize parameters
        #--------------------------------
        self.nvrs=self.xlo.shape[0]
        self.ntrgts=self.ylo.shape[0]
        self.ncstrs=self.glo.shape[0]
        
        self.fitnesslog_best=[]
        self.fitnesslog_worst=[]
        self.fitnesslog_ave=[]

        #--------------------------------
        # Initialize Population
        #--------------------------------
        if pop_init == None:
            pop_init=self.initialize_population()
        pop = pop_init[:]
        # -----------------------------
        # Iterate
        # -----------------------------
        for it in range(self.settingsdict["itmax"]):

            self.iter=it            
            pop=self.fun_evolve(pop)
            self.fun_print_ave_fitness(it)

        # -----------------------------
        # Plot Fitness
        # -----------------------------
        self.fitnesslog()
        print ("\t-> Optimization finished!")

    # =======================================================
    # Initialize random population in case no initial pop is given
    # =======================================================
    def initialize_population(self):
        pop = np.zeros((self.settingsdict["npop"],self.nvrs))
        pop[:,:] = (self.xlo + (self.xup-self.xlo)*np.random.rand(self.settingsdict["npop"],self.nvrs))
        return pop
    
    # =======================================================
    # Random mutate gen
    # =======================================================
    def fun_mutate_gene(self,array):
        mutateindex=np.random.randint(len(array))
        array[mutateindex]=self.xlo[mutateindex]+np.random.rand()*(self.xup[mutateindex]-self.xlo[mutateindex])
        return array

    # =======================================================
    # Evaluate Toolchain and feed penalty function
    # =======================================================
    def fun_eval_fitness(self,X):

        Y = np.zeros((self.pop_size,self.ntrgts))
        G = np.zeros((self.pop_size,self.ncstrs))
        # ----------------------------
        # Eval Function
        # ----------------------------
        if self.settingsdict["fitness_evalutation_mode"] == "parallel":
            Y[:,:] = self.fct(X)[0]
            G[:,:] = self.fct(X)[1]

        elif self.settingsdict["fitness_evalutation_mode"] == "serial":
            for n in range(X.shape[0]):
                feval = self.fct(X[n,:])
                Y[n,:],G[n,:] = feval[0],feval[1]


        return [Y,G]

    # =======================================================
    # Calculate penalty
    # =======================================================
    def fun_calc_penalty(self,Xbar,Ybar,Gbar):

        vals = [Xbar,Ybar,Gbar]
        P = np.zeros((Xbar.shape[0],1))

        for k in range(3):  # Check penalty for parameter, target and constraint space
            for n in range(Xbar.shape[0]):
                for p in range(vals[k].shape[1]):
                    error = np.abs(vals[k][n,p])-1
                    if error > 0.0:                        
                        # self.settingsdict["penalty_function"]
                        P[n,0]+=np.abs(error)     
        return P
    # =======================================================
    # Non dimensionlize dataset
    # =======================================================
    def fun_dim2dimless(self,W,wlo,wup):    # map to [-1,1] space
        Wbar = np.zeros(W.shape)
        for n in range(W.shape[0]):
            Wbar[n,:] = 2*(W[n,:]-wlo)/(wup-wlo)-1
        return Wbar
    # =======================================================
    # Dimensionlize dataset
    # =======================================================
    def fun_dimless2dim(self,Wbar,wlo,wup): # dim space
        W = np.zeros(Wbar.shape)
        for n in range(Wbar.shape[0]):
            W[n,:]=wlo+0.5*(Wbar[n,:]+1)*(wup-wlo)
        return W

    # =======================================================
    # Ranking manager
    # =======================================================
    def ranking_manager(self,Xbest,Ybest,PRbest):
    
        tresx = 0.01
        tresy = 0.01

        fac_pareto = self.settingsdict["fitness_manager"][0]
        fac_xdensity = self.settingsdict["fitness_manager"][1]
        fac_ydensity = self.settingsdict["fitness_manager"][2]
        fac_penalty = self.settingsdict["fitness_manager"][3]

        # -----------------------------------
        # 0 Rate points with small pareto rank higher
        # -----------------------------------
        prob_0 = 1.0/(1+fac_pareto*PRbest)
        #print(prob_0)
        # -----------------------------------
        # 1 Calculate design space density
        # -----------------------------------
        dist_X = self.niching_densityfunction(Xbest,disttres=tresx)
        x_neighbors = np.sum(dist_X,axis=1)
        prob_1 = 1.0/(1+fac_xdensity*x_neighbors)
        #print(prob_1)
        # -----------------------------------
        # 2 Calculate decision space density
        # -----------------------------------
        dist_Y = self.niching_densityfunction(Ybest,disttres=tresy)
        y_neighbors = np.sum(dist_Y,axis=1)
        prob_2 = 1.0/(1+fac_ydensity*y_neighbors)
        #print(prob_2)
        # -----------------------------------
        # 3 Penalty
        # -----------------------------------
        prob_3 = 1.0/(1+100*fac_penalty*Ybest[:,-1])
        #print(prob_3)
        # -----------------------------------
        # 4 global proba
        # -----------------------------------
        prob = prob_0*prob_1*prob_2*prob_3
        prob = prob/np.sum(prob)

        # -------------------------------------------------
        # Plot 
        # -------------------------------------------------
        showplot=False
        if self.iter % 10 == 0:
            cmap=plt.get_cmap('Reds')
            Ybest_dim = Ybest #self.fun_dimless2dim(Ybest[:,:-1],self.ylo,self.yup)

            for i in range(Ybest_dim.shape[1]):
                for j in range(Ybest_dim.shape[1]):
                    if i <= j:
                        for n in range(Ybest_dim.shape[0]):
                            if Ybest[n,-1]>0:
                                plt.plot(Ybest_dim[n,i],Ybest_dim[n,j],'s',color=cmap(prob[n]/np.max(prob)))
                            else:
                                plt.plot(Ybest_dim[n,i],Ybest_dim[n,j],'o',color=cmap(prob[n]/np.max(prob)))
                            #plt.plot(Ybest_dim[n,i]+tresy*np.cos(np.linspace(0,2*np.pi,30)),Ybest_dim[n,j]+tresy*np.sin(np.linspace(0,2*np.pi,30)),'k--',lw=0.5)
                            #plt.text(Ybest_dim[n,i],Ybest_dim[n,j],str(int(PRbest[n]))+"/"+str(int(x_neighbors[n]))+"/"+str(int(y_neighbors[n])),fontsize=8) 
      
                        plt.title("Target "+str(i)+" vs Target "+str(j))
                        plt.axis("equal")                      
                        axis=plt.axis()
                        plt.xlabel("Target "+str(i))
                        plt.ylabel("Target "+str(j))

                        plt.axvline(x=-1,color='r',linestyle = "--")
                        plt.axvline(x=1,color='r',linestyle = "--")
                        plt.axhline(y=-1,color='r',linestyle = "--")
                        plt.axhline(y=1,color='r',linestyle = "--")

                        plt.axis(axis)
                        plt.grid(True)

                        plt.savefig("plot_pareto_t"+str(i)+"_t"+str(j)+"_iter_"+str(self.iter)+".png", dpi=None, facecolor='w', edgecolor='w',
                        orientation='portrait', papertype=None, format="png",
                        transparent=False, bbox_inches='tight', pad_inches=0.1,
                        frameon=None)
                        plt.close()
        # -------------------------------------------------
        # Return probability for roulette wheel
        # -------------------------------------------------

        return prob

        
    # =======================================================
    # =======================================================
    # Evolution Operator
    # =======================================================
    # =======================================================
    def fun_evolve(self,pop):
    
        # -------------------------------------------------
        self.pop_size=pop.shape[0]
        
        pid = np.arange(self.iter*10,(self.iter+1)*10)
        # -------------------------------------------------
        # Eval fitness function....
        # -------------------------------------------------
        [fitness,constraints]=self.fun_eval_fitness(pop)

        # -------------------------------------------------
        # Load Paretopoints from previous iterations
        # -------------------------------------------------
        if os.path.isfile("globalbest_database"):
            data = np.loadtxt("globalbest_database")
            Idb=data[:,0]
            Xdb=data[:,1:self.nvrs+1]
            Ydb=data[:,self.nvrs+1:self.nvrs+self.ntrgts+1]
            Gdb=data[:,self.nvrs+self.ntrgts+1:self.nvrs+self.ntrgts+self.ncstrs+1]

            pop = np.append(Xdb[:],pop[:],axis=0)
            fitness =np.append(Ydb[:],fitness[:],axis=0)
            constraints =np.append(Gdb[:],constraints[:],axis=0)

            pid = np.append(Idb[:],pid[:],axis=0)
        # ---- ------------------------
        # Nondimenisionalize and calculate penalty
        # ----------------------------
        Ybar = self.fun_dim2dimless(fitness,self.ylo,self.yup)
        Gbar = self.fun_dim2dimless(constraints,self.glo,self.gup)
        Xbar = self.fun_dim2dimless(pop,self.xlo,self.xup)
        
        P = self.fun_calc_penalty(Xbar,Ybar,Gbar)
        # ----------------------------
        # Assemble Fitness-Penalty Vector
        # ----------------------------

        Ystar = np.append(Ybar[:],P[:],axis=1)
        
        # -----------------------------------
        # Calculate Paretoranks
        # -----------------------------------
        ranklimit = self.settingsdict["dbbest_paretorank"]
        
        pareto_ranks = np.asarray(calc_pareto_rank(Ystar))
        
        Ybest = Ystar[pareto_ranks<=ranklimit,:]
        Xbest = Xbar[pareto_ranks<=ranklimit,:]
        PRbest = pareto_ranks[pareto_ranks<=ranklimit]
  
        
        pop_filtered = pop[pareto_ranks<=ranklimit,:]
        fitness_filtered = fitness[pareto_ranks<=ranklimit,:]
        constraints_filtered = constraints[pareto_ranks<=ranklimit,:]
        
        pid_filtered = np.zeros((pop_filtered.shape[0],1))
        pid_filtered[:,0] = pid[pareto_ranks<=ranklimit]
        
        # -------------------------------------------------
        # Save filterd database
        # -------------------------------------------------
        database = np.append(pid_filtered[:],pop_filtered[:],axis=1)
        database = np.append(database[:],fitness_filtered[:],axis=1)
        database = np.append(database[:],constraints_filtered[:],axis=1)

        np.savetxt("globalbest_database",database)
        
        # -------------------------------------------------
        # Rank species
        # -------------------------------------------------
        pval = self.ranking_manager(Xbest,Ybest,PRbest)


        #-------------------------------------
        # Add to fitness log
        #-------------------------------------
        self.fitness_ave= np.mean(fitness_filtered,axis=0)
        self.fitness_best = np.min(fitness_filtered,axis=0)
        self.fitness_worst = np.max(fitness_filtered,axis=0)

        self.fitnesslog_best.append(self.fitness_best.tolist())
        self.fitnesslog_worst.append(self.fitness_worst.tolist())
        self.fitnesslog_ave.append(self.fitness_ave.tolist())

        #-------------------------------------
        # Create New Population!
        #-------------------------------------

        rankedpop=pop_filtered.tolist()

        self.bestmember=rankedpop[0]

        #-------------------------------------
        pop_next = []
        #-------------------------------------
        # Crossover
        #-------------------------------------
        desired_length = self.pop_size - len(pop_next)
        children = []
        drawnlist=[]

        #=========          
        while len(children) < desired_length:

            [male_index,female_index]=self.roulette_wheel_pop(2,pval)

            parentstag = str(male_index)+"_"+str(female_index)

            if male_index != female_index and not parentstag in drawnlist:
                
                male=rankedpop[male_index]
                female=rankedpop[female_index]
                
                drawnlist.append(parentstag)
                #=========
                if self.settingsdict["randomcrossover"]:
                    coi = self.nvrs/2 + np.random.randint(-self.nvrs/2,self.nvrs/2)
                else:
                    coi = self.nvrs/2 
                #=========
                if self.settingsdict["no_childs"] == 1 :
                    child1 = female[:coi] + male[coi:]
                    children.append(child1)
                elif self.settingsdict["no_childs"] == 2 :
                    child1 = female[:coi] + male[coi:]
                    child2 = male[:coi] + female[coi:]
                    children.append(child1)
                    children.append(child2)
        #=========          
        pop_next.extend(children[:desired_length])
        #=========          

        #-------------------------------------
        # Mutation
        #-------------------------------------
        # mutate some individuals
        for n in range(1,len(pop_next)):
            if self.settingsdict["mutate"] > np.random.rand():
                #pop_next[n]=self.fun_mutate(individual)
                pop_next[n]=self.fun_mutate_gene(pop_next[n])

        #-------------------------------------
        # Return new population
        #-------------------------------------
        return np.asarray(pop_next)
         
    # =======================================================
    # =======================================================
    # Share Function
    # =======================================================
    # =======================================================
    def niching_sharefunction(self,X,sigmatres=1.0):
        sh=np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            sh[i,i]=1
            for j in range(X.shape[0]):
                if i<j:
                    sigma=np.abs(np.mean(X[i,:])-np.mean(X[j,:]))

                    if sigma < sigmatres:
                        sh[i,j],sh[j,i]=1,1
        return sh

    # =======================================================
    # =======================================================
    # Density Function
    # =======================================================
    # =======================================================
    def niching_densityfunction(self,X,disttres=1.0):
        sh=np.zeros((X.shape[0],X.shape[0]))
        for i in range(X.shape[0]):
            for j in range(X.shape[0]):
                if i<j:
                    dist=(np.sum((X[i,:]-X[j,:])**2))**0.5
                    if dist < disttres:                     
                        sh[i,j],sh[j,i]=1,1
        return sh
    # =======================================================
    # =======================================================
    # Roulette Wheel
    # =======================================================
    # =======================================================
    def roulette_wheel_pop(self,nselect,probabilities):
        
        chosen = []
        while len(chosen)<nselect:
            r = np.random.random()
            probsum = 0.0
            for i in range(len(probabilities)):
                if r <= probsum and not i in chosen:
                    chosen.append(i)
                    break
                else:
                    probsum+=probabilities[i]
        return sorted(chosen)

    # =======================================================
    # =======================================================
    def fitnesslog(self):
        for k in range(self.ntrgts):

            plt.plot(np.asarray(self.fitnesslog_worst)[:,k],'r-')
            plt.plot(np.asarray(self.fitnesslog_ave)[:,k],'k-',lw=2)
            plt.plot(np.asarray(self.fitnesslog_best)[:,k],'b-', lw=2)

            plt.xlabel("Iterations")
            plt.ylabel("Fitness")
            
            plt.grid(True)

            plt.savefig("plot_cvg_target_"+str(k)+".png", dpi=None, facecolor='w', edgecolor='w',
            orientation='portrait', papertype=None, format="png",
            transparent=False, bbox_inches='tight', pad_inches=0.1,
            frameon=None)
            plt.close()
    # =======================================================
    def fun_print_ave_fitness(self,n):
    
        print("\t\t-> Iteration "+str(n))
        names=self.ntrgts*["Target"]
        names.append("Penalty")

        for n in range(self.ntrgts):
            ave = '%.3e' % self.fitness_ave[n]
            best = '%.3e' % self.fitness_best[n]
            worst = '%.3e' % self.fitness_worst[n]
            print("\t\t\t-> "+str(n)+" "+names[n]+":\t"+str(best)+"\t"+str(ave)+"\t"+str(worst))
    # =======================================================
    def fun_best_member(self):
        return np.asarray(self.bestmember)
    # =======================================================

#====================================================      
#====================================================
#====================================================
# Auxillary Functions
#====================================================
#====================================================
#====================================================





# =======================================================
# GET PARETO RANKING
# =======================================================
def calc_pareto_rank(Y):
    #------------------------------------------------
    for i in range(Y.shape[0]):
        for j in range(Y.shape[0]):
            if not i == j:
                if np.all(Y[i,:]==Y[j,:]):
                    Y[j,:]=np.ones(Y.shape[1])*1e+4
    #------------------------------------------------
    def dominates_check_1d(row, rowCandidate):
        return row <= rowCandidate 

    def dominates_check(row, rowCandidate):
        return all(r <= rc for r, rc in zip(row, rowCandidate))

    def cull(pts,pts_index, dominates):
        dominated = []
        dominated_index = []
        cleared = []
        cleared_index = []
        remaining = pts
        remaining_index = pts_index
        
        while remaining:
            candidate = remaining[0]
            candidate_index = remaining_index[0]
            new_remaining = []
            new_remaining_index = []
            
            for other,other_index in zip(remaining[1:],remaining_index[1:]):
                [new_remaining, dominated][dominates(candidate, other)].append(other)
                [new_remaining_index, dominated_index][dominates(candidate, other)].append(other_index)
                
            if not any(dominates(other, candidate) for other in new_remaining):
                cleared.append(candidate)
                cleared_index.append(candidate_index)
            else:
                dominated.append(candidate)
                dominated_index.append(candidate_index)
                
            remaining = new_remaining
            remaining_index = new_remaining_index

        return cleared_index, dominated,dominated_index
    #------------------------------------------------
    ntrgts = Y.shape[1]
    npop = Y.shape[0]

    dominated=Y[:].tolist()

    dominated_index=range(npop)
    rank=0
    paretoranks=npop*[""]
    
    while not dominated == []:

        [rankedpts, dominated,dominated_index]= cull(dominated,dominated_index, dominates_check)
        for rankedpt in rankedpts:
            paretoranks[rankedpt]=rank
        rank+=1
    #------------------------------------------------
    return paretoranks
    #------------------------------------------------

# =======================================================
# Rosenbrock Testfunction
# =======================================================
def testfct_rosenbrock(x):
    a,b=1,100
    z=(a-x[0])**2+b*(x[1]-x[0]**2)**2
    return [[z1],[]]
# =======================================================
# GoldsteinPrice Testfunction
# =======================================================
def testfct_goldsteinprice(x):
    
    z1=(1+(x[0]+x[1]+1)**2*(19-14*x[0]+3*x[0]**2-14*x[1]+6*x[0]*x[1]+3*x[1]**2))*(30+(2*x[0]-3*x[1])**2*(18-32*x[0]+12*x[0]**2+48*x[1]-36*x[0]*x[1]+27*x[1]**2))
    return [[z1],[]]

# =======================================================
# GoldsteinPrice Testfunction
# =======================================================
def testfct_binhkorn(x):
    
    z1=4*x[0]**2 + 4*x[1]**2
    z2=(x[0]-5)**2+(x[1]-5)**2

    g1 = (x[0]-5)**2+x[1]**2     #<=25
    g2 = (x[0]-8)**2+(x[1]+3)**2 # >=7.7
    return [[z1,z2],[g1,g2]]

#-------------------------------------
def testfct_contourplot(func,xlo,xup,nx=30,ny=30):

    XX,YY=np.meshgrid(np.linspace(xlo[0],xup[0],nx),np.linspace(xlo[1],xup[1],ny))
    ZZ=np.zeros((nx,ny))

    for i in range(nx):
        for j in range(ny):
            x = np.asarray([XX[i,j],YY[i,j]])
            ZZ[i,j]=func(x)[0][0]
    
    contourlevels=np.linspace(np.min(ZZ),np.max(ZZ),30)

    plt.contour(XX,YY,ZZ,contourlevels)








