#!/usr/bin/python
####################################################################################################################
import os

import module_ga

os.system("del plot_*")
os.system("del globalbest_*")
#==============================================
# Settings
#==============================================

#-------------------------------- 

#ylo,yup=[0],[2000]
#xlo,xup=[-2,-2],[2,2]
#glo,gup = [],[]
#func=module_ga.testfct_goldsteinprice

ylo,yup=[0,0],[2000,2000]
xlo,xup=[0,0],[5,3]
glo,gup = [0,7.7],[25,1e+10]
func= module_ga.testfct_binhkorn

#--------------------------------
# Optimizer Settings
#--------------------------------
settingsdict={}
settingsdict["niching_teshold"]=2.5
settingsdict["mutate"]=0.05
settingsdict["no_childs"]=1
settingsdict["randomcrossover"]=False
settingsdict["itmax"]=100
settingsdict["npop"]=10
settingsdict["2D_testfunction"]=False
settingsdict["verbose"]=True
settingsdict["2D_mo_testfunction"]=False
#--------------------------------
# Run the GA
#--------------------------------
print settingsdict
# ga=module_ga.genetic_algorithm(func,xlo,xup,ylo,yup,glo,gup,**settingsdict)

# print(ga.fun_best_member())




