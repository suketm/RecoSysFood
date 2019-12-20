#!/usr/bin/env python
# coding: utf-8

# import external libraries

import pandas as pd
import numpy as np
import random
import yaml
import os

# load external libraries

current_directory = os.getcwd()
os.chdir('../src/')

from utils.customer_acquisition_functions import CustomerAcquisition
from utils.customer_acquisition_functions import RecommenderSystem

os.chdir(current_directory)


# load config file

with open('..\\config\\config.yml', 'r', encoding='utf-8') as yml:
    config = yaml.load(yml, Loader=yaml.SafeLoader)


# constants

NUM_CUST_SIM = config['Cust_Acq']['NUM_CUST_SIM'] # No. of smilar customers, for example we use X(=10) most similar customers
NUM_FOOD_RECOMMENDED = config['Cust_Acq']['NUM_FOOD_RECOMMENDED'] # No. of recommended outlets


# load data
df_food_choice = pd.read_csv('..\\data\\'+config['File_Food_Choice'])

# parameters
num_transaction = df_food_choice.shape[0]
num_cust = df_food_choice['email'].nunique()
num_food = df_food_choice['food_choice'].nunique()
list_food = list(df_food_choice['food_choice'].unique())
arr_food = np.array(df_food_choice[['name_first', 'name_last', 'email', 'food_choice']])


def main():
    
    F = CustomerAcquisition()
    
    # Extract Data
    list_customer, list_customer_name_first, list_customer_name_last,     dict_cust_index, dict_cust_index_rev, dict_food_index, dict_food_index_rev,     arr_order_num_customer, arr_order_num_customer_bool = F.data_extraction(input_num_transaction = num_transaction, 
                                                                    input_num_cust = num_cust, 
                                                                    input_num_food = num_food,
                                                                    input_arr_food = arr_food, 
                                                                    input_list_food = list_food)
    
    ############ model: recommender system
    
    R = RecommenderSystem()
    
    # find similarirty score
    arr_similarity_score = R.similarity_score(input_arr_order_num_customer_bool = arr_order_num_customer_bool)
    
    # select similar customers for each customers, we select NUM_CUST_SIM = 10 most similar customers
    list_similar_customer = R.select_similar_customers(input_num_cust_sim = NUM_CUST_SIM, input_num_cust = num_cust,
                                                      input_arr_similarity_score = arr_similarity_score)
    
    # find expected visit for each store visited by similar customers
    arr_exp_order = R.exp_visits(input_num_cust = num_cust, 
                                 input_num_food = num_food,
                                 input_list_similar_customer = list_similar_customer,
                                 input_arr_order_num_customer = arr_order_num_customer,
                                 input_arr_similarity_score = arr_similarity_score)
    
    # find recommended outlet for each customer
    arr_recommendation = R.recommendation(input_num_cust = num_cust, input_num_food_recommended = NUM_FOOD_RECOMMENDED,
                                          input_arr_order_num_customer_bool = arr_order_num_customer_bool,
                                          input_num_food = num_food,
                                          input_arr_exp_order = arr_exp_order)
        
    # save recommendation
    R.recommendation_save(input_path = '..\\reports\\'+config['File_Recommendation'], input_file = arr_recommendation )
    
    ############ model: recommender system example

    # some example to test recommendation for customer
    idx_customer = random.randint(0,num_cust-1)
    R.recommendation_example(input_idx_customer = idx_customer, input_list_customer_name_first = list_customer_name_first,
                             input_list_customer_name_last = list_customer_name_last,
                             input_arr_recommendation = arr_recommendation,
                             input_dict_food_index_rev = dict_food_index_rev)


if __name__ == '__main__':
    main()

