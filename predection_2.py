#!/bin/sh

import itertools
from collections import defaultdict
import random
import numpy as np
from multiprocessing import Pool
from multiprocessing.pool import ThreadPool

import pickle
import multiprocessing
from pandarallel import pandarallel
import pandas as pd 
import pickle
from tqdm.notebook import tqdm
from flip_pancake import pancake_neighbours
import ast
from slide_tile import slide_neighbours
import math
from itertools import repeat

def h(state):
    return sum(abs(state[i]-state[i-1])>1 for i in range(1,len(state)))

def p_and_h(state):

    state= np.random.permutation(state)
    return h(state)


def permutation_and_h(n,n_limit=100):
    h_dict=defaultdict(int)
    init_list= list(range(1,n+1))
    # res_list= list(map(lambda x: sum(abs(x[i]-x[i-1])>1 for i in range(1,len(x))), [list(x)for x in itertools.permutations(init_list)]))
    # per_list= list(itertools.islice(itertools.permutations(init_list), 3,8))
    # per_list=[ np.random.permutation(init_list) for i in range(n_limit)]

    res_list= list(pool.map(p_and_h, repeat(init_list,n_limit)))
    for i in res_list:
        h_dict[i]+=1

    total_h= sum(h_dict.values())        
    for i in range(0,n+1):
        if i >=1:
            h_dict[i]+=h_dict[i-1]
    for i in range(0,n+1):
        h_dict[i]=h_dict[i]/total_h
    return h_dict



def calcalute_KRE(d,problem_size):
    p=all_dict_h[problem_size]
    b=problem_size-1
    sum_res=0
    for i in range(0,d+1):
        sum_res+=b**i*p[d-i]
    return sum_res

def cacalute_per_state_h(state):
    # state=list(np.random.permutation(state))
    lst=[]
    h_state=h(state)
    for neighbor in pancake_neighbours(list(state)):
        neighbor=neighbor[1]
        h_neighbor= h(neighbor)
        lst.append(h_neighbor)
    return h_state,lst
        

def condisinal_h(n,n_limit=100):
    dict_h={}
    init_list= list(range(1,n+1))
    for i in range (n):
        dict_h[i]=defaultdict(int)
    res_lst=[ list(np.random.permutation(init_list)) for i in range(n_limit)]
    # res_lst=[ list(np.random.permutation(init_list)) for i in range(n_limit)]
    res_lst= pool.map(cacalute_per_state_h, res_lst)

    # res_lst= pool.map(cacalute_per_state_h, repeat(init_list,n_limit))
    # for father_h, neighbours_h_lst in pool.map(cacalute_per_state_h, repeat(init_list,n_limit)):
    for father_h, neighbours_h_lst in res_lst:
        for h_neighbor in neighbours_h_lst:
            dict_h[father_h][h_neighbor]+=1
       
    for key in dict_h.keys():
        total=sum(dict_h[key].values())
        for key2 in dict_h[key].keys():
            dict_h[key][key2]=dict_h[key][key2]/total
    return dict_h


#check number of cores in your machine


# pandarallel.initialize(progress_bar=True)
# from multiprocesspandas import applyparallel

#read csv 


def helper(x):
    return calcalute_KRE(x["optimal_cost"],x["problem_size"])


# pandarallel.initialize(progress_bar=True)
from multiprocesspandas import applyparallel


def calcalute_CDP(s,d):
    sum_res=0
    for i in range (0,d+1):
        for v in range (0,d-i+1):
            sum_res+=calcalute_CDP_helper(i,s,v,d)
    return sum_res


def calcalute_CDP_helper(i,s,v,d):
    state_h=h(s)
    b=len(s)-1
    problem_size=len(s)
    local_dict=all_dict_condisonal_h[problem_size]
    
    if i==0 and v==state_h:
        return 1 
    if i==0 and v!=state_h:
        return 0
    else:
        sum_res=0
        for v_p in range (0,d-(i-1)+1):
            try:
                if state_h not in local_dict:
                    p=0
                else:
                    p=local_dict[state_h][v_p]
            except keyError:
                p=0
            if p!=0:
                sum_res+=calcalute_CDP_helper(i-1,s,v_p,d)*b*p #Todo: check the order of the dict
    return sum_res


def helper_cdp(x):
    return calcalute_CDP(ast.literal_eval(x["start"]),x["optimal_cost"])

import os
if __name__ == '__main__':
    pandarallel.initialize(progress_bar=False)

    print(multiprocessing.cpu_count())


    pool=Pool(multiprocessing.cpu_count())
    # pool=ThreadPool(32)
    # print(1)
    n=10
    # n_limit=int(math.factorial(n)*0.01)
    # n_limit=int(math.factorial(n)*0.01)
    # n_limit=5000000
    n_limit=int(math.factorial(n))
    

    print("number of sample seach state is: "+str(n_limit))
    # print(2)

    df_res = pd.read_csv('pancake_results_'+str(n)+'.csv')
    all_dict_h={}

    print("start calculating all_dict_h")
    if os.path.exists(f'all_dict_h_{n}.pickle'):
        with open(f'all_dict_h_{n}.pickle', 'rb') as handle:
            all_dict_h = pickle.load(handle)
    else:
        for problem_size in df_res.problem_size.unique():
            all_dict_h[problem_size]=permutation_and_h(problem_size,n_limit=n_limit)
        # save to pickle
        with open(f'all_dict_h_{n}.pickle', 'wb') as handle:
            pickle.dump(all_dict_h, handle, protocol=pickle.HIGHEST_PROTOCOL)


    print("start calculating pred_KRE")
    tqdm.pandas()
    df_res["pred_KRE"]= df_res.progress_apply(helper,axis=1)
    df_res.sort_values(by=['optimal_cost'],inplace=True)



    print("start calculating all_dict_condisonal_h")
    all_dict_condisonal_h={}
    for problem_size in df_res.problem_size.unique():
        all_dict_condisonal_h[problem_size]=condisinal_h(problem_size,n_limit=n_limit)


    # save to pickle
    with open(f'all_dict_condisonal_h_{n}.pickle', 'wb') as handle:
        pickle.dump(all_dict_condisonal_h, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("start calculating pred_CDP")
    pool.close()

    # calcalute_CDP(ast.literal_eval(df_res.iloc[0]["start"]),df_res.iloc[0]["optimal_cost"])
    # df_res["pred_CDP"]= df_res.progress_apply(helper_cdp,axis=1)

    df_res["pred_CDP"]= df_res.parallel_apply(helper_cdp,axis=1)


    df_res.to_csv("res_cdp_"+str(n)+".csv",index=False)
