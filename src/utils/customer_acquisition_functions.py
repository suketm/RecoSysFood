
### 1. IMPORTS
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances
from scipy.spatial.distance import cosine

### 2. CLASSES

class CustomerAcquisition(object):

	def __init__(self):
		pass

	def data_extraction(self, input_num_transaction, input_num_cust, 
			input_num_food, input_arr_food, input_list_food):

		'''
		Function: Extract Data
		Arguement:
			input_num_transaction				: total number of transaction
			input_num_cust						: total number of customers
			input_num_food						: number of food choices
			input_list_food						: list of food choices
			input_arr_food						: transactional data
		Output:
			output_list_customer 				: list of customers
			output_list_customer_name_first 	: list of first name of customers
			output_list_customer_name_last 		: list of first name of customers
			output_dict_cust_index 				: dictionary with keys name of customers, value = index of that customer
			output_dict_cust_index_rev			: dictionary with keys index of customers, value = name of customer
			output_dict_food_index 				: dictionary with keys name of food, value = index of that outlet
			output_dict_food_index_rev	 		: dictionary with keys index of food, value = food name
			ourput_arr_order_num_customer 	 	: graph: customer vs. num of order for a food
			output_arr_order_num_customer_bool	: graph: customer vs. food order (bool)

		'''
		

		output_list_customer = []
		output_list_customer_name_first = []
		output_list_customer_name_last = []
		output_dict_cust_index = {}
		output_dict_cust_index_rev = {}
		output_dict_food_index  = {}
		output_dict_food_index_rev = {}
		output_arr_order_num_customer = np.zeros((input_num_cust, input_num_food))
		output_arr_order_num_customer_bool = np.zeros((input_num_cust, input_num_food))
		
		for transaction in range(input_num_transaction):

			if input_arr_food[transaction,2] not in output_list_customer:
				output_list_customer.append(input_arr_food[transaction,2])
				output_list_customer_name_first.append(input_arr_food[transaction,0])
				output_list_customer_name_last.append(input_arr_food[transaction,1])

		output_dict_cust_index = {output_list_customer[idx_customer]:idx_customer for idx_customer in range(input_num_cust)}
		output_dict_cust_index_rev = dict (zip(output_dict_cust_index.values(),output_dict_cust_index.keys()))

		output_dict_food_index = {input_list_food[idx_food]:idx_food for idx_food in range(input_num_food)}
		output_dict_food_index_rev = dict (zip(output_dict_food_index.values(),output_dict_food_index.keys()))

		for transaction in range(input_num_transaction):

			output_arr_order_num_customer[ output_dict_cust_index[input_arr_food[transaction,2]],
										 output_dict_food_index[input_arr_food[transaction,3]] ] += 1

			output_arr_order_num_customer_bool[ output_dict_cust_index[input_arr_food[transaction,2]],
										 output_dict_food_index[input_arr_food[transaction,3]] ] = 1

		return (output_list_customer, output_list_customer_name_first, output_list_customer_name_last,
				output_dict_cust_index, output_dict_cust_index_rev, output_dict_food_index, 
				output_dict_food_index_rev, output_arr_order_num_customer, output_arr_order_num_customer_bool)


class RecommenderSystem (object):

	def __init__(self):
		pass

	def similarity_score(self, input_arr_order_num_customer_bool):

		'''
		Function:
		Argument:
			input_arr_order_num_customer_bool	: graph: customer vs. food order (bool)
		Output:
			output_arr_similarity_score			: graph: similarirty score among different customers
		'''

		output_arr_similarity_score = 1 - pairwise_distances(input_arr_order_num_customer_bool, metric="cosine")

		return output_arr_similarity_score


	def select_similar_customers(self, input_num_cust_sim, input_num_cust, input_arr_similarity_score):

		'''
		Function: select similar customers for each customers
		Argument:
			input_num_cust_sim				: number of similar customers to select
			input_num_cust 					: total number of customers
			input_arr_similarity_score		: graph: similarirty score among different customers
		Output:
			output_list_similar_customer	: list of similar customer
		'''

		output_list_similar_customer = []

		for customer in range(input_num_cust):
			
			# we take it to -1, since each element is most simillar to itself
			output_list_similar_customer.append(list(np.argsort(
										input_arr_similarity_score[customer,:])[-input_num_cust_sim-1:-1]
										))

		return output_list_similar_customer


	def exp_visits(self, input_num_cust, input_num_food, input_list_similar_customer,
					input_arr_order_num_customer, input_arr_similarity_score):

		'''
		Function: find expected visit for each store visited by similar customers
		Argument:
			input_num_cust 					: total number of customers
			input_num_food					: number of food
			input_list_similar_customer		: list of similar customer 
			input_arr_order_num_customer 	: graph: customer vs. num of order for a food
			input_arr_similarity_score		: graph: similarirty score among different customers
		Output:
			output_arr_exp_order			: graph: customers vs. expected order
		'''

		output_arr_exp_order = np.zeros((input_num_cust, input_num_food))

		for idx_customer in range(input_num_cust):
			
			for similar_customer in input_list_similar_customer[idx_customer]:
				output_arr_exp_order[idx_customer,:] += input_arr_order_num_customer[similar_customer] \
										*float(input_arr_similarity_score[idx_customer,similar_customer] \
											/sum(input_arr_order_num_customer[similar_customer,:]))


		return output_arr_exp_order


	def recommendation(self, input_num_cust, input_num_food_recommended, 
			input_arr_order_num_customer_bool, input_num_food, input_arr_exp_order):

		'''
		Function: find recommended outlet for each customer
		Arguments:
			input_num_food_recommended			: number of recommendations
			input_num_cust 						: number of customers
			input_arr_order_num_customer_bool 	: graph: customer vs. visit in outlet (bool)
			input_num_food 						: number of outlets
			input_arr_exp_order					: graph: customers vs. expected visit in outlets
		Output:
			output_arr_recommendation 			: recommendations for each customer
		'''
		output_arr_recommendation = np.zeros((input_num_cust, input_num_food_recommended))

		for idx_customer in range(input_num_cust):

			temp_list_recommendation_customer = input_arr_exp_order[idx_customer,:]
			temp_list_idx_recommendation_customer = \
							sorted(range(len(temp_list_recommendation_customer)), 
								key=lambda k: temp_list_recommendation_customer[k],
								reverse = True)
			
			for idx_outlet in range(input_num_food):

				if input_arr_order_num_customer_bool[idx_customer,idx_outlet] != 0:

					temp_list_idx_recommendation_customer.remove(idx_outlet)

			output_arr_recommendation[idx_customer, :] = \
				temp_list_idx_recommendation_customer[:input_num_food_recommended]

		return output_arr_recommendation


	def recommendation_example(self, input_idx_customer, input_list_customer_name_first,
								input_list_customer_name_last, input_arr_recommendation,
								input_dict_food_index_rev):
		"""
		Function: prints the result for some (one) customer
		Argument:
			input_idx_customer				: index of some customer
			input_list_customer_name_first	: list of first name of customers
			input_list_customer_name_last	: list of last name of customers
			input_arr_recommendation		: recommendations for each customer
			input_dict_food_index_rev		:  dictionary with keys index of food, value = outlet
		Output:
			None
		"""
		print ("For customer",input_list_customer_name_first[input_idx_customer]+' '+\
			input_list_customer_name_last[input_idx_customer]," recommended foods are:",)
		for outlet in input_arr_recommendation[input_idx_customer,:]:
			print (input_dict_food_index_rev[outlet])

		return None


	def recommendation_save(self, input_path , input_file):

		"""
		Function: saves a pickle file
		Argument:
			input_path		: path where to save
			input_file		: file to save	
		Output:
			Saves pickle file
		"""
		pd.DataFrame(input_file).to_pickle(input_path)