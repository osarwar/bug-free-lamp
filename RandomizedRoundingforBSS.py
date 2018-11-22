# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 19:23:31 2018

@author: osarwar
"""

from pyomo.environ import *
from pyomo.opt import SolverFactory
from copy import copy 
import time
import numpy as np 
import scipy as sp 
import math
from math import log
from numpy import random
from collections import OrderedDict
import os 
import matplotlib.pyplot as plt

#Import R functions 
import rpy2.robjects as ro
from rpy2.robjects.packages import importr
d = {'package.dependencies': 'package_dot_dependencies',
     'package_dependencies': 'package_uscore_dependencies'}
bestsubset_tibshirani = importr('bestsubset', robject_translations=d)
scale = ro.r['scale']
cv_glmnet = ro.r['cv.glmnet']
coef = ro.r['coef']
predict = ro.r['predict']
as_matrix = ro.r['as.matrix']

class BuildDataArrays_and_OptimizationModels(): 
    
    def __init__(self, n, p, rho=0, sparsity_pattern=1, beta_type=1,
                 snr=1, n_val=0, n_test=50): 
        
        self.n = n 
        self.p = p 
        
        self.n_test = n_test
        
        #Generate synthetic data according to tibshirani best_subset package 
        Data_test_train = bestsubset_tibshirani.sim_xy(n,p,n_test,rho,sparsity_pattern,beta_type,snr)
        Data_val = bestsubset_tibshirani.sim_xy(n_val,p,0,rho,sparsity_pattern,beta_type,snr)
        
        self.x_train_R = scale(Data_test_train[0])
        self.y_train_R = scale(Data_test_train[1])
        self.x_test_R = scale(Data_test_train[2])
        self.y_test_R = scale(Data_test_train[3])
        self.x_val_R = scale(Data_val[0])
        self.y_val_R = scale(Data_val[1])
        
        self.x_train = np.array(scale(Data_test_train[0]))
        self.y_train = np.array(scale(Data_test_train[1]))
        self.x_test = np.array(scale(Data_test_train[2]))
        self.y_test = np.array(scale(Data_test_train[3]))
        self.x_val = np.array(scale(Data_val[0]))
        self.y_val = np.array(scale(Data_val[1]))
    
#        
#    def construct_data_arrays(self):
#        
#        self.Atrain = self.x_train
#        self.btrain = self.y_train 
#        self.Atest = self.x_test
#        self.btest = self.y_test
        
    
    def construct_MIQP(self, x, y, complexity_penalty, complexity_penalty_type, bigM): 

        regressors = [r for r in range(1,x.shape[1]+1)]
        datapoints = [d for d in range(1,x.shape[0]+1)]
        
        self.MIQP = ConcreteModel()
        self.MIQP.Coeff = Var(regressors, domain=Reals)
        self.MIQP.z = Var(regressors, domain=Binary)   
        self.MIQP.V = Var(datapoints, domain=Reals)
        
        def ub_rule(model,i):
            return model.Coeff[i] <= float(bigM)*model.z[i]
        def lb_rule(model,i):
            return model.Coeff[i] >= -float(bigM)*model.z[i]
        def obj_rule(model,i): 
            return model.V[i] == (float(y[i-1])-sum(model.Coeff[j]*float(x[i-1][j-1]) for j in regressors))
        
        self.MIQP.UB = Constraint(regressors, rule=ub_rule)  
        self.MIQP.LB = Constraint(regressors, rule=lb_rule)
        self.MIQP.Vconst = Constraint(datapoints,rule=obj_rule)
        
        if complexity_penalty_type == 'BIC': 
            self.MIQP.OBJ = Objective(expr=len(datapoints)*log(sum((self.MIQP.V[i])**2 for i in datapoints)/len(datapoints)) + complexity_penalty*sum(self.MIQP.z[i] for i in regressors))
        else: 
            self.MIQP.OBJ = Objective(expr=sum((self.MIQP.V[i])**2 for i in datapoints) + complexity_penalty*sum(self.MIQP.z[i] for i in regressors))
            
        self.opt = SolverFactory('gurobi')
        
        return self.MIQP, self.opt 


    def construct_QP(self, x, y, complexity_penalty, complexity_penalty_type, bigM): 
        
        regressors = [r for r in range(1,x.shape[1]+1)]
        datapoints = [d for d in range(1,x.shape[0]+1)]
        
        self.QP = ConcreteModel()
        self.QP.Coeff = Var(regressors, domain=Reals)
        self.QP.z = Var(regressors, domain=UnitInterval)   
        self.QP.V = Var(datapoints, domain=Reals)
        
        def ub_rule(model,i):
            return model.Coeff[i] <= float(bigM)*model.z[i]
        def lb_rule(model,i):
            return model.Coeff[i] >= -float(bigM)*model.z[i]
        def obj_rule(model,i): 
            return model.V[i] == (float(y[i-1])-sum(model.Coeff[j]*float(x[i-1][j-1]) for j in regressors))
        
        self.QP.UB = Constraint(regressors, rule=ub_rule)  
        self.QP.LB = Constraint(regressors, rule=lb_rule)
        self.QP.Vconst = Constraint(datapoints,rule=obj_rule)
        
        if complexity_penalty_type == 'BIC': 
            self.QP.OBJ = Objective(expr=len(datapoints)*log(sum((self.QP.V[i])**2 for i in datapoints)/len(datapoints)) + complexity_penalty*sum(self.QP.z[i] for i in regressors))
        else: 
            self.QP.OBJ = Objective(expr=sum((self.QP.V[i])**2 for i in datapoints) + complexity_penalty*sum(self.QP.z[i] for i in regressors))
        
        self.opt = SolverFactory('gurobi',executable=r"C:\gurobi801\win64\bin\gurobi.bat")
    
        return self.QP, self.opt 
            

class LinAlgandUpdates(): 
    
    def __init__(self, x, y):
        
        self.x = x 
        self.regressors_old_A = [1 for i in range(self.x.shape[1])]
        self.regressors_old_QR = [1 for i in range(self.x.shape[1])]
        self.Q, self.R = sp.linalg.qr(self.x)
        self.A = x 
        self.b = y 
    
    def updateA_col(self): 
        
        h = 0 
        for i in range(self.x.shape[1]): 
            if self.regressors_old_A[i] == 0 and self.regressors[i] == 1 : #New variable inserted, inserts corresponding column into A  
                self.A = np.insert(self.A.T, h, self.x.T[i],0)
                self.A = self.A.T
                h = h + 1 
            if self.regressors_old_A[i] == 1 and self.regressors[i] == 1: 
                h = h + 1 
            if self.regressors_old_A[i] == 1 and self.regressors[i] == 0: #Variable removed, deletes corresponding column from A  
                self.A = np.delete(self.A.T,h,0)
                self.A = self.A.T
    
    def updateQR(self):
        
        h = 0 
        for i in range(self.x.shape[1]): 
            if self.regressors_old_QR[i] == 0 and self.regressors[i] == 1 : #New variable inserted, inserts corresponding column into A  
                self.Q, self.R = sp.linalg.qr_insert(self.Q,self.R,self.x.T[i].T,h,'col')
                h = h + 1 
            if self.regressors_old_QR[i] == 1 and self.regressors[i] == 1: 
                h = h + 1 
            if self.regressors_old_QR[i] == 1 and self.regressors[i] == 0: #Variable removed, deletes corresponding column from A  
                self.Q, self.R = sp.linalg.qr_delete(self.Q,self.R,h,1,'col')
        
    
    def OLS_soln(self): 

        self.updateA_col()
        if np.linalg.matrix_rank(self.A) == self.A.shape[1]: 
            self.updateQR()
            Rp = self.R[:np.count_nonzero(self.regressors)] #Takes the first 'p' rows of R 
            nb = np.dot(self.Q.T,self.b)
            c = nb[:np.count_nonzero(self.regressors)] #Takes the first 'p' rows of nb vector 
            d = nb[np.count_nonzero(self.regressors):]
            self.B_ols = sp.linalg.solve_triangular(Rp, c)
            self.SSRols = sum(d[i]**2 for i in range(np.shape(self.A)[0]-np.shape(self.A)[1]))
            self.B_ols_sum = sum(abs(self.B_ols[i]) for i in range(np.shape(self.A)[1]))
            
            self.regressors_old_A = copy(self.regressors)
            self.regressors_old_QR = copy(self.regressors)
        else: 
            self.B_ols, self.SSRols, rank, s = np.linalg.lstsq(self.A,self.b,rcond=-1)
            self.B_ols_sum = sum(abs(self.B_ols[i]) for i in range(self.A.shape[1]))
            if len(self.SSRols) == 0: 
                self.SSRols = 0
            else: 
                self.SSRols = self.SSRols[0]
            self.regressors_old_A = copy(self.regressors)    
    
    def evaluate_obj(self, regressors, complexity_penalty, complexity_penalty_type, n):
        
        self.regressors = regressors 
        self.complexity_penalty = complexity_penalty 
        self.OLS_soln()
        if complexity_penalty_type == 'BIC' and self.SSRols != 0: 
            self.obj = n*log(self.SSRols/n) + self.complexity_penalty*np.count_nonzero(self.regressors)
        else: 
            self.obj = self.SSRols + self.complexity_penalty*np.count_nonzero(self.regressors)
        
        if self.obj != 0:  
            self.obj = self.obj[0]
        return self.obj, self.B_ols, self.B_ols_sum
        
        
         
        
        
class RandomizedRounding(): 
    
    def __init__(self, file, DM, normalize_bvs=True,
                 complexity_penalty_type='CrossVal', bigMoptions=['B_ols_sum',2], cv_folds=5, benchmark=False,
                 benchmark_time=300): 
        
        self.DM = DM
        self.LAU = LinAlgandUpdates(x=self.DM.x_train, y=self.DM.y_train)
        self.normalize_bvs = normalize_bvs
        self.bigMoptions = bigMoptions 
        self.benchmark = benchmark
        self.benchmark_time = 60 
        self.keepfiles = False 
        if self.benchmark: 
            self.benchmark_time = benchmark_time
            self.keepfiles = False
        self.complexity_penalty_type = complexity_penalty_type 
        if self.complexity_penalty_type == 'BIC': 
            self.complexity_penalty = log(DM.n)
        elif self.complexity_penalty_type == 'CrossVal': 
            MaxLambda = 2*np.linalg.norm(DM.x_train.T@DM.y_train, ord=np.inf)
            self.complexity_penalty = np.linspace(MaxLambda/15, MaxLambda, 10)
            self.cv_folds = cv_folds
        else:  
            self.complexity_penalty = self.complexity_penalty_type*np.linalg.norm(DM.x_train.T@DM.y_train, ord=np.inf)
        
        print('Complexity Penalty:', self.complexity_penalty, file=results)
        _, B_ols, self.B_ols_sum = self.LAU.evaluate_obj(np.ones(DM.p), 0, self.complexity_penalty_type, self.DM.n)
        self.maxBols = abs(max(B_ols,key=abs)) 
        
    def optimize(self, opt, model): 
        
        opt.options['timelimit'] = self.benchmark_time
        results = opt.solve(model, tee=False, keepfiles=self.keepfiles)
        self.solve_time = results.solver.time 
        self.keepfiles=False #only keep files for initial benchmark 
        regressors = []
        coefficients = []
        for i in range(1,self.DM.p+1): 
            regressors.append(value(model.z[i]))
            coefficients.append(value(model.Coeff[i]))
            
        return np.array(regressors), np.array(coefficients), results.solver.time
    
    def normalize_prob(self, regressors_probability):
        
        maxProb = max(regressors_probability)
        minProb = min(regressors_probability)
        for i in range(len(regressors_probability)): 
            regressors_probability[i] = (regressors_probability[i] - minProb)/(maxProb-minProb)
        return regressors_probability
    
    def rand_round(self, complexity_penalty, regressors_probability):
        
        large_bigM_normalize_bvs = False 
        while True: 
            if self.normalize_bvs: 
                regressors_probability = self.normalize_prob(regressors_probability)
            if large_bigM_normalize_bvs: 
                regressors_probability = self.normalize_prob(regressors_probability)
            opt_regressors = np.zeros(len(regressors_probability))
            opt_coeffs = np.zeros(len(regressors_probability))
            opt_obj = 10000000
            for rr in range(self.num_rr): 
                regressors= np.ones(len(regressors_probability)) 
                for i in range(len(regressors)): 
                    regressors[i] = np.random.choice([0,1], p=[1-regressors_probability[i],regressors_probability[i]])
                if np.count_nonzero(regressors) > 0 and np.count_nonzero(regressors) < self.DM.n:
                    obj, coeff, _ = self.LAU.evaluate_obj(regressors, complexity_penalty, self.complexity_penalty_type, self.DM.n)
                    if obj < opt_obj: 
                        opt_regressors = copy(regressors)
                        opt_coeffs = copy(coeff)
                        opt_obj = copy(obj)
            if np.count_nonzero(opt_regressors) == 0: 
                large_bigM_normalize_bvs = True #in case bigM too big and thus probabilites are too small 
            else: 
                break 
        #include zero coefficients in opt_coeff array     
        opt_coeffs_all = np.zeros(len(regressors_probability))
        j = 0 
        for i in range(len(opt_regressors)): 
            if opt_regressors[i] == 1: 
                opt_coeffs_all[i] = opt_coeffs[j]
                j+=1 

        return opt_regressors, opt_coeffs_all, opt_obj
        
    def cross_val(self): 
        
        opt_error = 10000
        opt_complexity_penalty = 1000
        size_folds = floor(self.DM.n/cv_folds)
        
        for complexity_penalty in self.complexity_penalty: 
            x = self.DM.x_train
            y = self.DM.y_train 
            x_test = []
            y_test = []
            RMSE_per_fold = []
            for fold in range(cv_folds): 
                for j in range(1,size_folds+1): #leave out one fold 
                        x = np.delete(x,(fold*size_folds),0)
                        y = np.delete(y,(fold*size_folds))
                        x_test.append(x[fold*size_folds])
                        y_test.append(y[fold*size_folds])
                np.array(x_test)
                np.array(y_test)
                QP, opt = self.DM.construct_QP(x, y, complexity_penalty=complexity_penalty, complexity_penalty_type=self.complexity_penalty_type, bigM=self.bigM)
                regressors_probability, _, QPtime,_ = self.optimize(opt, QP)
                self.LAU = LinAlgandUpdates(x, y)
                _, opt_coeffs, _= self.rand_round(x, y, regressors_probability)
                RMSE_per_fold.append(self.test_fit(self.DM.x_test, self.DM.y_test, self.opt_coeffs))
            average_error_for_complexity_penalty = sum(RMSE_per_fold)/len(RMSE_per_fold) 
            if  average_error_for_complexity_penalty < opt_error: 
                opt_error = average_error_for_complexity_penalty 
                opt_complexity_penalty = complexity_penalty 
        
        self.complexity_penalty = opt_complexity_penalty
        #regress using optimal penalty          
        QP, opt = self.DM.construct_QP(x, y, complexity_penalty=opt_complexity_penalty, complexity_penalty_type=self.complexity_penalty_type, bigM=self.bigM)
        regressors_probability, _, QPtime = self.optimize(opt, QP)
        self.LAU = LinAlgandUpdates(self.DM.x_train, self.DM.y_train)
        self.opt_regressors, self.opt_coeffs, self.opt_coeffs = self.rand_round(x, y, regressors_probability)
        
    def minBIC_or_ArbitraryComplexityPenalty(self): 

        QP, opt = self.DM.construct_QP(self.DM.x_train, self.DM.y_train, complexity_penalty=self.complexity_penalty, complexity_penalty_type=self.complexity_penalty_type, bigM=self.bigM)
        regressors_probability, _, QPtime = self.optimize(opt, QP)
        self.LAU = LinAlgandUpdates(self.DM.x_train, self.DM.y_train)
        self.opt_regressors, self.opt_coeffs, self.opt_obj = \
        self.rand_round(self.complexity_penalty, regressors_probability)
        
    def test_fit(self, xtest, ytest, Coeffs): 
        
        SquaredError = sum((ytest[point] - sum(Coeffs[regressor]*xtest[point][regressor] for regressor in range(len(Coeffs))))**2 for point in range(xtest.shape[0]))
        MSE = SquaredError/xtest.shape[0]
        RMSE = (MSE)**(0.5)
        return RMSE[0] 
    
    def plot(self, n_list, p_list, rho, sparsity_pattern, snr, beta_type, benchmark_obj_array, benchmark_error_array, 
             lasso_obj_array, lasso_error_array, fwdss_obj_array, fwdss_error_array, RR_obj_array_array, RR_error_array_array, 
             RR_FSS_obj_array, RR_FSS_error_array, filepath): 
        
        old_obj_array = np.array([np.empty(len(p_list))])
        old_obj_array[:] = np.nan

        old_obj_array = np.append(old_obj_array, np.array([benchmark_obj_array, lasso_obj_array, fwdss_obj_array, RR_FSS_obj_array]), axis=0)
        obj_array = np.append(old_obj_array, RR_obj_array_array.T, axis=0)
        
        old_error_array = np.array([np.empty(len(p_list))])
        old_error_array[:] = np.nan
        
        old_error_array = np.append(old_error_array, np.array([benchmark_error_array, lasso_error_array, fwdss_error_array, RR_FSS_error_array]), axis=0)
        error_array = np.append(old_error_array, RR_error_array_array.T, axis=0)
        
        
        min_obj_array = np.nanmin(obj_array, axis=0)
        min_error_array = np.nanmin(error_array, axis=0)
        
        obj_array_normalized = obj_array/min_obj_array 
        error_array_normalized = error_array/min_error_array 
        
        plt.figure(figsize=(18,12))
        plt.plot(p_list, obj_array_normalized[1], linestyle = '-.' , marker='o', color = 'b', label='Benchmark')
        plt.plot(p_list, obj_array_normalized[2], linestyle = '--', marker='o', color = 'm', label='Lasso')
        plt.plot(p_list, obj_array_normalized[3], linestyle = ':', marker='o', color = 'g', label='Forward SS') 
        plt.plot(p_list, obj_array_normalized[4], linestyle = ':', marker='o', color = 'c', label='RR FSS')
        j = 0 
        h = 0 
        for i in range(len(RR_obj_array_array.T)): 
            exec("plt.plot(p_list, obj_array_normalized[5+%d], marker='*', label='RR_M=%s_nrr=%d')" % (i,self.bigMoptions[j],self.number_randomized_rounds_list[h]))
            h+=1
            if h == (len(self.number_randomized_rounds_list)): 
                h = 0
                j+=1
             
        plt.ylabel('OBJ/BestOBJ')
        plt.xlabel('Problem size (# Regressors)')
        plt.title('Scaled Objective Value, rho=' + str(rho) + ' sparsity_pattern=' + str(sparsity_pattern) + ' snr=' + str(snr) \
                  + ' beta_type=' + str(beta_type))
        if True in (obj_array_normalized[2] >= 2):
            plt.yscale('log', basey=2)
        plt.legend()
        plt.savefig(filepath + '\objective_plots.png')
        
        plt.figure(figsize=(18,12))
        plt.plot(p_list, error_array_normalized[1],  marker='o', color = 'b', linestyle = '-.', label='Benchmark')
        plt.plot(p_list, error_array_normalized[2],  marker='o', color = 'm', linestyle = '--', label='Lasso')
        plt.plot(p_list, error_array_normalized[3],  marker='o', color = 'g', linestyle = ':', label='Forward SS') 
        plt.plot(p_list, error_array_normalized[4], linestyle = ':', marker='o', color = 'c', label='RR FSS')
        j = 0 
        h = 0 
        for i in range(len(RR_obj_array_array.T)): 
            exec("plt.plot(p_list, error_array_normalized[5+%d],  marker='*', label='RR_M=%s_nrr=%d')" % (i,self.bigMoptions[j],self.number_randomized_rounds_list[h]))
            h+=1 
            if h == (len(self.number_randomized_rounds_list)): 
                h = 0
                j+=1
        plt.ylabel('Error/BestError')
        plt.xlabel('Problem size (# Regressors)')
        plt.title('Scaled RMSE Value, rho=' + str(rho) + ' sparsity_pattern=' + str(sparsity_pattern) + ' snr=' + str(snr) \
                  + ' beta_type=' + str(beta_type))
        plt.legend()
        plt.savefig(filepath + '\error_plots.png')
        
        
        print('SUMMARY RESULTS==================================================================================', file=results)
        print('Benchmark - # Best OBJ, AVG Scaled OBJ, # Best Error, AVG Scaled Error:', np.count_nonzero((obj_array_normalized[1]>=1) & (obj_array_normalized[1]<=1.025)), np.nanmean(obj_array_normalized[1]), \
              np.count_nonzero((error_array_normalized[1]>=1) & (error_array_normalized[1]<=1.025)), np.nanmean(error_array_normalized[1]), file=results)
        print('Lasso - # Best OBJ, AVG Scaled OBJ, # Best Error, AVG Scaled Error:', np.count_nonzero((obj_array_normalized[2]>=1) & (obj_array_normalized[2]<=1.025)), np.nanmean(obj_array_normalized[2]), \
              np.count_nonzero((error_array_normalized[2]>=1) & (error_array_normalized[2]<=1.025)), np.nanmean(error_array_normalized[2]), file=results)
        print('FSSel - # Best OBJ, AVG Scaled OBJ, # Best Error, AVG Scaled Error:', np.count_nonzero((obj_array_normalized[3]>=1) & (obj_array_normalized[3]<=1.025)), np.nanmean(obj_array_normalized[3]), \
              np.count_nonzero((error_array_normalized[3]>=1) & (error_array_normalized[3]<=1.025)), np.nanmean(error_array_normalized[3]), file=results)
        print('RR FSS - # Best OBJ, AVG Scaled OBJ, # Best Error, AVG Scaled Error:', np.count_nonzero((obj_array_normalized[4]>=1) & (obj_array_normalized[4]<=1.025)), np.nanmean(obj_array_normalized[4]), \
              np.count_nonzero((error_array_normalized[4]>=1) & (error_array_normalized[4]<=1.025)), np.nanmean(error_array_normalized[4]), file=results)
        j, h = 0, 0 
        for i in range(len(RR_obj_array_array.T)): 
            exec("print('RR M=%s nrr=%d - # Best OBJ, AVG Scaled OBJ, # Best Error, AVG Scaled Error:', np.count_nonzero((obj_array_normalized[5+%d]>=1) & (obj_array_normalized[5+%d]<=1.025)), np.mean(obj_array_normalized[5+%d]), np.count_nonzero((error_array_normalized[5+%d]>=1) & (error_array_normalized[5+%d]<=1.025)), np.mean(error_array_normalized[5+%d]), file=results)" % (self.bigMoptions[j],self.number_randomized_rounds_list[h], i, i, i, i, i, i))
            h+=1 
            if h == (len(self.number_randomized_rounds_list)): 
                h = 0
                j+=1
    
    def RR(self, number_randomized_rounds_list = [100]):
        
        self.number_randomized_rounds_list = number_randomized_rounds_list 
        
        #Exact solution to MIQP             
        if self.benchmark and self.complexity_penalty_type != 'CrossVal': 

            self.MIQP, opt = self.DM.construct_MIQP(x=self.DM.x_train, y=self.DM.y_train, complexity_penalty=self.complexity_penalty, complexity_penalty_type=self.complexity_penalty_type, bigM=self.B_ols_sum)
            bench_y, bench_coeff, bench_time = self.optimize(opt, self.MIQP)
            self.RMSE_bench = self.test_fit(self.DM.x_test, self.DM.y_test, bench_coeff)


            print('BENCHMARK RESULTS=============================================================', file=results)
            print('Benchmark Solve Time:', self.solve_time, file=results)
            print('Benchmark Regressors & Number:', np.nonzero(bench_coeff)[0], len(np.nonzero(bench_coeff)[0]),file=results)
            print('Benchmark Coefficients:', bench_coeff[np.nonzero(bench_coeff)[0]], file=results)
            print('Benchmark OBJ:', self.MIQP.OBJ(), file=results)
            print('Benchmark RMSE:', self.RMSE_bench, file=results)
           
            
        #Randomized rounding approximate solution
        if plot: 
            self.RR_obj_list = []
            self.RR_error_list = []
            
        print('RANDOMIZED ROUNDING RESULTS====================================================', file=results)
        for bigM in self.bigMoptions: 
            
            #set Big-M 
            if bigM == 'B_ols_sum': 
                self.bigM = self.B_ols_sum
            else: 
                self.bigM = bigM*self.maxBols
            print('+++bigM:', self.bigM[0], file=results)
            
            #Perform randomized rounding 
            for self.num_rr in self.number_randomized_rounds_list: 
                
                print('# Randomized Rounds:', self.num_rr, file=results)
                      
                if self.complexity_penalty_type == 'CrossVal': 
                    self.cross_val()
                else: 
                    self.minBIC_or_ArbitraryComplexityPenalty()
                
                
                self.OBJ_RR = self.opt_obj
                self.RMSE_RR = self.test_fit(self.DM.x_test, self.DM.y_test, self.opt_coeffs)
                
                print('QP Solve Time:', self.solve_time, file=results)
                print('RR Regressors & Number:', np.nonzero(self.opt_coeffs)[0], len(np.nonzero(self.opt_coeffs)[0]), file=results)
                print('RR Coefficients:', self.opt_coeffs[np.nonzero(self.opt_coeffs)[0]], file=results)
                print('RR OBJ:', self.OBJ_RR, file=results)
                print('RR RMSE:', self.RMSE_RR, file=results)
                
                if plot: 
                    self.RR_obj_list.append(self.OBJ_RR)
                    self.RR_error_list.append(self.RMSE_RR)

    def lasso(self): 
        
        LassoSolution = cv_glmnet(self.DM.x_train_R,self.DM.y_train_R,standardize=False, intercept=False)  
        h = coef(LassoSolution,s="lambda.min")
        h = as_matrix(h)
        LassoCoefSum = 0 
        LassoRegressors = []
        LassoCoeffs = []
        for i in range(self.DM.p): 
            if h[i] != 0: 
                LassoRegressors.append(i)
                LassoCoeffs.append(h[i])
            LassoCoefSum += abs(h[i])
        LassoPredictions_train = predict(LassoSolution,self.DM.x_train_R,"lambda.min")
        LassoPredictions_test = predict(LassoSolution,self.DM.x_test_R,"lambda.min")
        #compute lasso corresponding to BSS objective
        SE_lasso = 0
        j = 0 
        for i in LassoPredictions_train: 
            SE_lasso += (self.DM.y_train_R[j]-i)**2 
            j+=1  
        if self.complexity_penalty_type == 'BIC':
            self.OBJ_lasso = self.DM.n*log(SE_lasso/self.DM.n) + self.complexity_penalty*len(LassoRegressors)
        else: 
            self.OBJ_lasso = SE_lasso + self.complexity_penalty*len(LassoRegressors)
        #compute lasso RMSE 
        SE_lasso = 0
        j = 0 
        for i in LassoPredictions_test: 
            SE_lasso += (self.DM.y_test_R[j]-i)**2 
            j+=1   
        self.RMSE_lasso = (SE_lasso/j)**(0.5)
        print('LASSO RESULTS==================================================================', file=results)
        print('Lasso Regressors & Number:', LassoRegressors, len(LassoRegressors), file=results)
        print('Lasso Coefficients:', LassoCoeffs, file=results)
        print('Lasso OBJ:', self.OBJ_lasso, file=results )
        print('Lasso RMSE:', self.RMSE_lasso, file=results)
            
    def fwdss(self):
        
        
        self.NumLSPS_FSS = 0 
        opt_obj = 1000000
        opt_coeffs = np.zeros(self.DM.p)
        FSsoln = bestsubset_tibshirani.fs(self.DM.x_train_R, self.DM.y_train_R,intercept=False)
        for i in range(1,min(self.DM.p,2000)): 
            predictfss = bestsubset_tibshirani.predict_fs(FSsoln,self.DM.x_train_R,i)
            SE = 0 
            self.NumLSPS_FSS += (self.DM.p + 1 - i) 
            for j in range(self.DM.n): 
                SE += (self.DM.y_train_R[j] - predictfss[j])**2
            if self.complexity_penalty_type == 'BIC':
                obj = self.DM.n*log(SE/self.DM.n) + self.complexity_penalty*i
            else: 
                obj = SE + self.complexity_penalty*i 
            if obj < opt_obj: 
                opt_obj = copy(obj)
                opt_coeffs= bestsubset_tibshirani.coef_fs(FSsoln,i)
            else:  
                break 
            
        FwdSSRegressors = []
        FwdSSCoeffs = []
        for i in range(self.DM.p): 
            if opt_coeffs[i] != 0.0: 
                FwdSSRegressors.append(i)
                FwdSSCoeffs.append(opt_coeffs[i])
                
        self.OBJ_fss = opt_obj
        
        #calculate RMSE
        predictfss = bestsubset_tibshirani.predict_fs(FSsoln,self.DM.x_test_R,len(FwdSSRegressors))
        SE_fss = 0 
        for j in range(self.DM.n_test): 
            SE_fss += (self.DM.y_test_R[j] - predictfss[j])**2
        self.RMSE_fss = (SE_fss/self.DM.n_test)**(0.5)
        
        
        print('FwdSS RESULTS==================================================================', file=results)
        print('FwdSS Regressors & Number:', FwdSSRegressors, len(FwdSSRegressors), file=results)
        print('FwdSS Coefficients:', FwdSSCoeffs, file=results)
        print('FwdSS OBJ:', self.OBJ_fss, file=results)
        print('FwdSS RMSE:', self.RMSE_fss, file=results)
        print('FwdSS # LSTSQ Probs Solved:', self.NumLSPS_FSS, file=results)
    
    def RR_FSS(self, number_RR_FSS): 
        
        QP, opt = self.DM.construct_QP(self.DM.x_train, self.DM.y_train, complexity_penalty=self.complexity_penalty, complexity_penalty_type=self.complexity_penalty_type, bigM=self.bigM)
        regressors_probability, _, QPtime = self.optimize(opt, QP)
#        self.LAU = LinAlgandUpdates(self.DM.x_train, self.DM.y_train)
        
        
        regressors_probability_sorted = np.sort(regressors_probability)[::-1]
        regressors_sorted = np.argsort(regressors_probability)[::-1]
        
        self.NumLSPS_RR_FSS = 0 
        opt_obj = 1000000000
        for n in range(number_RR_FSS): 
            regressors= np.zeros(len(regressors_probability)) 
            i = 0 
            step_obj = 100000000000
            step = True 
            while step: 
                select = np.random.choice([0,1], p=[1-regressors_probability_sorted[i],regressors_probability_sorted[i]])
                if select == 1: 
                    regressors[regressors_sorted[i]] = 1 
                    obj, coeff, _ = self.LAU.evaluate_obj(regressors, self.complexity_penalty, self.complexity_penalty_type, self.DM.n)
                    self.NumLSPS_RR_FSS += 1 
                    if obj < step_obj: 
                        step_obj = copy(obj)
                        step_coeffs = copy(coeff)
                        step_regressors = copy(regressors) 
                    else:
                        step = False
                i += 1 
                if i == len(regressors): 
                    step = False 
            
            if step_obj < opt_obj: 
                opt_obj = copy(step_obj)
                opt_coeffs = copy(step_coeffs)
                opt_regressors = copy(step_regressors)
            
        #include zero coefficients in opt_coeff array     
        opt_coeffs_all = np.zeros(len(regressors_probability))
        j = 0 
        for i in range(len(opt_regressors)): 
            if opt_regressors[i] == 1: 
                opt_coeffs_all[i] = opt_coeffs[j]
                j+=1 
        
        self.OBJ_RR_FSS = opt_obj
        self.RMSE_RR_FSS = self.test_fit(self.DM.x_test, self.DM.y_test, opt_coeffs_all)
        
        
        print('RR FSS RESULTS==================================================================', file=results)
        print('RR FSS Regressors & Number:', np.nonzero(opt_coeffs)[0], len(np.nonzero(opt_coeffs)[0]), file=results)
        print('RR FSS Coefficients:', np.ndarray.flatten(opt_coeffs[np.nonzero(opt_coeffs)[0]]), file=results)
        print('RR FSS OBJ:', self.OBJ_RR_FSS, file=results)
        print('RR FSS RMSE:', self.RMSE_RR_FSS, file=results)
        print('RR FSS # LSTSQ Probs Solved:', self.NumLSPS_RR_FSS, file=results)
        
        
        
        
        
def main(file, n_list, p_list, rho, sparsity_pattern,
         snr, beta_type, n_test, n_val, number_randomized_rounds_list, bigMoptions, complexity_penalty_type, 
         cv_folds, normalize_bvs, test_lasso, test_forwardstepwise, test_RR_FSs, number_RR_FSS,
         benchmark, benchmark_time, max_p_benchmark, plot, filepath):

    if plot:
        
        benchmark_obj_list = [] #to keep track of obj for every problem tested
        benchmark_error_list = []  
        lasso_obj_list = []
        lasso_error_list = []
        fwdss_obj_list = []
        fwdss_error_list = []
        RR_FSS_obj_list = []
        RR_FSS_error_list = []
        
        RR_obj_list_list = [] #to keep track of obj/error for every problem for list of every RR configuration (in terms of BM and #RRs)
        RR_error_list_list = []
        
        NumLSPS_FSS_List = []
        NumLSPS_RR_FSS_List = []

    for p in p_list: 
        n = n_list[p_list.index(p)]
        
        print('++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++', file=results)
        print('PROBLEM SIZE============================================================', file=results)
        print('DataPoints:', n, 'TotalRegressors:', p, file=results)
        
        if p > max_p_benchmark: 
            benchmark = False
        
        DM = BuildDataArrays_and_OptimizationModels(n, p, 
                    rho, sparsity_pattern, beta_type,
                    snr, n_val, n_test)
        RandRound = RandomizedRounding(file, DM, normalize_bvs,
                     complexity_penalty_type, bigMoptions, cv_folds, 
                     benchmark, benchmark_time)
        
        RandRound.RR(number_randomized_rounds_list)
        
        
        
        if test_lasso: 
            
            RandRound.lasso()
            
            if plot: 
                
                lasso_obj_list.append(RandRound.OBJ_lasso)
                lasso_error_list.append(RandRound.RMSE_lasso)
            
        if test_forwardstepwise: 
            
            RandRound.fwdss()
            
            NumLSPS_FSS_List.append(RandRound.NumLSPS_FSS)
            
            if plot: 
                
                fwdss_obj_list.append(RandRound.OBJ_fss)
                fwdss_error_list.append(RandRound.RMSE_fss)
                
        if test_RR_FSS: 
            
            RandRound.RR_FSS(number_RR_FSS)
            
            NumLSPS_RR_FSS_List.append(RandRound.NumLSPS_RR_FSS)
            
            if plot: 
                
                RR_FSS_obj_list.append(RandRound.OBJ_RR_FSS)
                RR_FSS_error_list.append(RandRound.RMSE_RR_FSS)

        
        if plot and benchmark and complexity_penalty_type != 'CrossVal':
            
            benchmark_obj_list.append(copy(RandRound.MIQP.OBJ()))
            benchmark_error_list.append(RandRound.RMSE_bench)
            
        if plot: 
            
            RR_obj_list_list.append(RandRound.RR_obj_list)
            RR_error_list_list.append(RandRound.RR_error_list)
            
        if plot and benchmark==False: 
            
            benchmark_obj_list.append(np.nan)
            benchmark_error_list.append(np.nan)
            
        if plot and test_lasso==False: 
            
            lasso_obj_list.append(np.nan)
            lasso_error_list.append(np.nan)
            
        if plot and test_forwardstepwise==False: 
            
            fwdss_obj_list.append(np.nan)
            fwdss_error_list.append(np.nan)
            
        if plot and test_RR_FSS==False: 
            
            RR_FSS_obj_list.append(np.nan)
            RR_FSS_error_list.append(np.nan)
            
        
    
    if plot: 
        
        RandRound.plot(n_list, p_list, rho, sparsity_pattern, snr, beta_type, np.array(benchmark_obj_list), np.array(benchmark_error_list), np.array(lasso_obj_list), np.array(lasso_error_list),
             np.array(fwdss_obj_list), np.array(fwdss_error_list), np.array(RR_obj_list_list), np.array(RR_error_list_list),
             np.array(RR_FSS_obj_list), np.array(RR_FSS_error_list), filepath)
        
    if test_forwardstepwise: 
        print('FSS AVG # LSTSQ Problems Solved:', np.mean(NumLSPS_FSS_List), file=results)
    
    if test_RR_FSS: 
        print('RR FSS AVG # LSTSQ Problems Solved:', np.mean(NumLSPS_RR_FSS_List), file=results)
                

    
if __name__ == '__main__':
    
    test_code = False
    
    n_test = 60
    n_val = 0 
    number_randomized_rounds_list = [300]
    normalize_bvs = False  ##boolean
    bigMoptions = [3.5]
    complexity_penalty_type = 0.1  ##or "CrossVal" or "BIC" or "float" value in float*\\X.T*Y\\_inf 
    cv_folds = 5
    
    test_lasso = False
    test_forwardstepwise = True 
    test_RR_FSS = True 
    number_RR_FSS = 100 
    benchmark = False #boolean true or false
    benchmark_time = 300
    max_p_benchmark = 1000 
    plot = True
    

    n_list = [int(n) for n in np.linspace(100,400,25).astype(int)]
    p_list = [int(p) for p in np.linspace(100,2500,25).astype(int)]
    
    if test_code: 
        n_list = [5, 30, 60, 50]
        p_list = [8, 50, 100, 1000]
    
    rho_options = [0.3]
    sparsity_pattern_options = [10]
    snr_options = [0.5]
    beta_type_options = [1]
    
    for rho in rho_options: 
        for sparsity_pattern in sparsity_pattern_options: 
            for snr in snr_options: 
                for beta_type in beta_type_options: 
                    
                    time_string = time.strftime('%Y%m%d_%H%M')
                    
#                    os.mkdir("Results\AIChETalk")
                    
                    filepath = r"Results\EXP" \
                    + time_string  + 'rho_' + str(rho)  +'snr_' + str(snr) \
                    + 's_' + str(sparsity_pattern) + 'b_' + str(beta_type)
                    
                    os.mkdir(filepath)
                    parameters = open(filepath + '\parameters.txt', 'a')
                    
                    
                    print('EXPERIMENT INFORMATION=============================================================', file=parameters)
                    print('rho', rho, 'sparsity_pattern', sparsity_pattern, 'snr', snr, 'beta_type', beta_type, file=parameters)
                    print('n_list', n_list, 'p_list', p_list, file=parameters)
                    print('n_val', n_val, '# RR options', number_randomized_rounds_list, file=parameters)
                    print('normalize rounding probabilities', normalize_bvs, file=parameters)
                    print('bigMoptions', bigMoptions, 'compelxity_penalty_type', complexity_penalty_type, 'cv_folds', cv_folds, file=parameters) 
                    print('test_lasso', test_lasso, 'test_forward_stepwise', test_forwardstepwise, 'test_RR_FSS', test_RR_FSS, '# RR_FSS', number_RR_FSS, file=parameters)
                    print('benchmark', benchmark, 'benchmark_time', benchmark_time, file=parameters) 
                    parameters.close()
                    
                    results = open(filepath +'\data.txt', 'a')
                    
                    main(results, n_list, p_list, rho, sparsity_pattern,
                         snr, beta_type, n_test, n_val, number_randomized_rounds_list, bigMoptions, complexity_penalty_type, 
                         cv_folds, normalize_bvs, test_lasso, test_forwardstepwise, test_RR_FSS, number_RR_FSS,
                         benchmark, benchmark_time, max_p_benchmark, plot, filepath)
                    
                    results.close()
    