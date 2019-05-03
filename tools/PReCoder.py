#!/usr/bin/env python
# coding: utf-8

# # PReCoder notebook
# ##### Transforming a fact table in RDS format (containing a mapping of descriptional hierarchical info on facts) into a fact table on P-code level <p>
# Differences with legacy version of the PCode algorithm:
# - Rewritten for Python 3 instead of 2.7
# - Main function pcode_file() has to be called for each input file separately. In the old version a dictionary containing file names of multiple inputs could be fed
# - All configuration of the input file (columns etc.) has been removed from the function and has to be done in the extract phase using a metadata tables setup
# - The function can handle a different number (and also missing levels) of admin levels present in the facts table and the locations table
# - As a result of the above mentioned points, one standardized input file at a time can be processed by the more standardized PReCoder
# - Deleted a lot of obsolete code
# - Added a lot of comment

# #### Import necessary libraries

# In[1]:


import psycopg2
import sys, os
import numpy as np
import pandas as pd
import pandas.io.sql as psql
from os import listdir
from os.path import isfile, join
import difflib
import re


# #### Set up connection with Postgres database

# In[2]:


# In the current script the database connection is not being used. During development we load csv-type files instead 
conn_string = "host = rabobankcra.postgres.database.azure.com                port = 5432               dbname = cra_dwh               user = Megatron@rabobankcra               password = Rabobank2019"

conn = psycopg2.connect(conn_string)


# #### Read standardized pcode-locations table from csv

# In[3]:


# This is a temporary work around. We think pcode-locations table should be generated using a metadata driven piece of code
pcode_loc = pd.read_csv('../../input/pcode_loc_std.csv', delimiter = ';')


# In[4]:


# Show how pcode-locations table looks like
pcode_loc.head()


# #### Read facts table from csv

# In[14]:


# Read facts table
# Should be parameterized!!
facts = pd.read_csv('../../input/RDS_UGA_02_UGA_20140101.dta', delimiter = '|')


# In[15]:


facts


# In[6]:


# Rename columns in the current facts table. Should be done outside this script. Preferably in the metadata configuration.
# In this example the L3 admin level in the facts table is missing. Script below can deal with this, but of course the right
# admin levels should be assigned to the proper columns in the source table
facts.rename(columns={'Country': 'L0_name',
                      'CountryGeoLevel_01': 'L1_name',
                      'CountryGeoLevel_02': 'L2_name',
                      'CountryGeoLevel_03': 'L4_name',
                      'CountryGeoLevel_04': 'L3_name',
                      'CountryGeoLevel_05': 'L5_name'}, inplace=True)


# In[7]:


# Convert column names to object type
facts[['L0_name', 'L1_name', 'L2_name', 'L3_name', 'L4_name', 'L5_name']] = facts[['L0_name', 'L1_name', 'L2_name', 'L3_name', 'L4_name', 'L5_name']].astype(object)


# In[8]:


# Add new columns to be filled with p-codes later on
facts['L0_code'] = np.nan
facts['L1_code'] = np.nan
facts['L2_code'] = np.nan
facts['L3_code'] = np.nan
facts['L4_code'] = np.nan
facts['L5_code'] = np.nan


# In[9]:


# Show how facts table looks like
facts.head()


# #### Configure Pcoder

# In[10]:


# SPECIFY INTERACTION PARAMETERS

# specify the threshold score below which an input from the user is requested
ask_below_score =  {'L0':0.9, 'L1':0.9, 'L2':0.9, 'L3':0.8, 'L4':0.8, 'L5':0.8}
# if the score is below the reject level it is considered not found 
reject_below_score= {'L0':0.9, 'L1':0.9, 'L2':0.9, 'L3':0.55, 'L4':0.55, 'L5':0.55}


# In[11]:


# SPECIFY INTERACTION PARAMETERS
# Copy of the above specification, but parameters set such that no user input is asked

# specify the threshold score below which an input from the user is requested
ask_below_score =  {'L0':0.9, 'L1':0.9, 'L2':0.9, 'L3':0.55, 'L4':0.55, 'L5':0.55}
# if the score is below the reject level it is considered not found 
reject_below_score= {'L0':0.9, 'L1':0.9, 'L2':0.9, 'L3':0.55, 'L4':0.55, 'L5':0.55}


# In[12]:


# Definitions of necessary p-code functions

# Respect to the original developer of the p-code algorithm:
# Code developed by Marco Velliscig (marco.velliscig AT gmail.com)
# for the dutch Red Cross
# released under GNU GENERAL PUBLIC LICENSE Version 3

# Main function, to be called to start p-coding
def pcode_file(df_loc, df_facts, sav_name, ask_below_score , reject_below_score, known_matches={}, name_tricks = True) :
        
        # Only (admin) level names that are present in the locations input table are extracted
        # e.g. ['L0', 'L1', 'L4']
        level_tag = [col[0:2] for col in df_loc.columns.values if re.findall("L.*_name", col)]
        level_tag.sort()
        level_tag_name = [level + '_name' for level in level_tag]
        
        for level in level_tag_name:
                # Convert all values in facts and locations tables to uppercase
                df_loc[level]=df_loc[level].str.upper()
                df_facts.loc[df_facts[level].notnull(), level] = df_facts.loc[df_facts[level].notnull(), level].str.upper()
                
                # Per admin level create a new column (later on to be filled with the best match)
                df_facts[level[0:2] + '_best_match_name'] = np.NaN 

        #### a simple join is tried first for exact matches
        # Legacy code!
        # For now we commented out the exact match, because exact match needs a fixed number of columns.
        # Our idea is to be flexible in the number of admin levels to be filled. Note that for different facts
        # (e.g. rainfall and number of hospitals) the lowest admin level can be different.

        #df_match = pd.merge(df_facts, df_loc, on = level_tag_name, how = 'left')

        # Forward and backward pass
        df_match = match_against_template(df_facts, df_loc, level_tag, ask_below_score, reject_below_score, reverse = False)
        df_match = match_against_template(df_facts, df_loc, level_tag, ask_below_score, reject_below_score, reverse = True)

        # Saving the known matches so they can be loaded and modified
        # later version the known matches file can be specified in the options
        # warm_start
        # By default known_matches is an empty dict

        df_known_matches = pd.DataFrame.from_dict(known_matches, orient='index')
        #df_known_matches.reset_index()
        #df_known_matches.columns = [ 'name_raw' , 'name_match' ]

        # Write output file to csv
        # Later on to be inserted in database table (staging.riskindicator_facts)
        df_match.to_csv(sav_name) #,encoding='windows-1251')

        print(' list of no matches' , sum(df_match[level_tag[-1]+'_code'].isnull()))
        print(' list of matches ' , sum(df_match[level_tag[-1]+'_code'].notnull()))

        
def construct_known_match_tag(name, upper_level):
        """ Function that creates a tag with the name and the previous level
        to be added to the list of known matches
        """
        if (not isinstance(name, str)) | ( not isinstance(upper_level, str)) : print(name , upper_level)
        return str(name) + ' ' + upper_level

    
def find_best_match_user_input( poss_matches , 
                                name_to_match,  
                                upper_level , 
                                score_threshold, 
                                reject_threshold, 
                                known_matches , 
                                use_tricks=False):
        """ record linkage function that selects from a list of candidates name
        the best match for a given name applying string metrics 
        a list of known matches is also passed
        thresholds can be specified
        """
        
        # Seems not to be relevant on the deepest level?
        known_match_tag = construct_known_match_tag(name_to_match, upper_level)

        try :
                # Try if the above constructed tag is present in the known_matches dictionary
                best_match = known_matches[known_match_tag]
        except :
                if use_tricks :
                    # Trim the strings from words like city and capital that can reduce 
                    # the accuracy of the match
                    poss_matches_trim = [poss_matches[i].replace('CITY','').replace('OF','').replace('POB.','').strip() for i in range(len(poss_matches))]
                    
                    for i in range(len(poss_matches_trim)):
                        if len(poss_matches_trim[i]) > 9 :
                            if poss_matches_trim[i][-9] == 'POBLACION':
                                poss_matches_trim[i] = poss_matches_trim[i].replace('POBLACION','').strip()

                    regex = re.compile(".*?\((.*?)\)")
                    poss_matches_trim = [re.sub("[\(\[].*?[\)\]]", "", poss_matches_trim[i]) for i in range(len(poss_matches))]
                    poss_matches_trim = [poss_matches_trim[i].strip() for i in range(len(poss_matches))]
                    name_to_match_trim = name_to_match.replace('CITY','').replace('OF','').strip()
                    if len(name_to_match_trim) > 9 :
                        if name_to_match_trim[-9:] == 'POBLACION':
                            name_to_match_trim = name_to_match_trim.replace('POBLACION','').strip()
                    name_to_match_trim = re.sub("[\(\[].*?[\)\]]", "", name_to_match_trim)
                    name_to_match_trim =      name_to_match_trim.strip()               

                else:
                    poss_matches_trim = poss_matches
                    name_to_match_trim = name_to_match

                # Use logic from difflib package to do fuzzy matching
                ratio = [(difflib.SequenceMatcher(None,poss_matches_trim[i], name_to_match_trim)).ratio()                          for i in range(len(poss_matches))]
                
                # List containing all possibilities with their score
                vec_poss = list(zip(poss_matches, ratio))
                vec_poss_sorted = np.array(sorted(vec_poss, key=lambda x: x[1], reverse=True))
                try: 
                        # Select most probable match
                        most_prob_name_match = vec_poss_sorted[0,0]

                except:
                        print('error')
                        print('name to match ', name_to_match)
                        print('poss matches' , poss_matches)
                        most_prob_name_match  = 'error'
                        return most_prob_name_match

                # Select ratio of most probable match
                best_ratio = vec_poss_sorted[0,1]

                if float(best_ratio) <= reject_threshold:
                        most_prob_name_match  = 'Not found'
                        
                elif (float(best_ratio) > reject_threshold) &                      (float(best_ratio) < score_threshold): 
                     # Ask if the possible match is right
                        print('is ' , most_prob_name_match , 'the right match for ' , name_to_match , '(score:',best_ratio , ')')
                        respond = input('press return for yes, everything else for no : ')

                        if respond != '' : 
                                sorted_prob_name_match =vec_poss_sorted[:,0]
                                sorted_prob_name_match_numbered = np.array(zip(sorted_prob_name_match, range(len(sorted_prob_name_match))))
                                print('\n select from the best match for ' ,name_to_match ,' from this list: \n',  sorted_prob_name_match_numbered)

                                while True : 
                                        selected_index = input('select the right choice by number, press return for not found : ')
                                        if selected_index == '' :
                                                most_prob_name_match  = 'Not found'
                                                break

                                        elif selected_index.isdigit():
                                                most_prob_name_match = sorted_prob_name_match_numbered[int(selected_index),0]
                                                break
                                        else:
                                                continue
                                                
                # Update the known matched dictionary
                known_matches[known_match_tag] = most_prob_name_match 
                print('==' , most_prob_name_match , 'is the right match for ' , name_to_match , best_ratio , '\n')
                best_match=most_prob_name_match

        return best_match 


def match_against_template(df_facts, df_loc, level_tag, ask_below_score, reject_below_score, exception = [], reverse = False, verbose = False, name_tricks = True):

        # the code usually does the matching from the shallower to the deeper level
        # but it can also go the other way around even if it is less efficient this way
        # if you combine the 2 approaches you should account for most cases
        known_matches = {}
        counter = 0
        n_perfect_matches = 0 
        n_no_matches = 0 
        if reverse : 
                # Remember: level_tag originates in the location table and looks like ['L0', 'L1', 'L4']
                level_tag_use = list(reversed(level_tag))
        else:
                level_tag_use = level_tag 

        # Loop over all rows in the facts table where the deepest admin level is null
        for index in  df_facts.loc[df_facts[level_tag[-1]+'_code'].isnull()].index :

                df_loc_matches = df_loc
                upper_level = ''
                for admin_level in level_tag_use :
                        if verbose : 
                                print('len template dataframe level', admin_level                                        , len(df_loc_matches))
                                print(df_loc_matches.describe())
                        
                        # Gets the name of the admin level for the index entry
                        name_admin_level = df_facts.loc[index][admin_level+'_name']

                        # If name_admin_level is in the exceptions list, then continue to the next iteration 
                        # Also, if one of the values in the facts table is nan: continue
                        if (name_admin_level in exception) | (pd.isnull(name_admin_level)) : continue
                        # It tries to get a perfect match straight away
                        # !!!! this is not needed if a match is made by merge first
                        
                        # Counts number of times that an exact match is found between the selected value
                        # in the facts table and the locations table (for the corresponding admin level)
                        n_matches_current_level = sum(df_loc_matches[admin_level + '_name'] == 
                                              name_admin_level)
                        if verbose : print('num matches', admin_level,  n_matches_current_level)
                


                        if (n_matches_current_level) > 0 :
                                if verbose : print('')

                        elif (n_matches_current_level) == 0 :
                                print("perc completed ", ((float(counter) / len(df_facts.index)) * 100), '\n')
                                
                                # Select all unique values for the corresponding admin level in the locations table
                                poss_matches = (df_loc_matches[admin_level + '_name'].drop_duplicates()).values
                                score_threshold = ask_below_score[admin_level]     
                                reject_threshold = reject_below_score[admin_level]
                                #print(index, upper_level)

                                best_match = find_best_match_user_input(poss_matches, name_admin_level, upper_level, score_threshold, reject_threshold, known_matches, use_tricks = name_tricks) 
                                if best_match == 'Not found' :  
                                       n_no_matches +=1 
                                       #print('************* Not found, doing full search **********')
                                       print(df_facts.loc[index])
                                       #add here the full search instead 
                                       
                                       #break 
                                elif best_match == 'error' :         
                                       n_no_matches +=1 
                                       print('************* error admin ', admin_level, name_admin_level)
                                       print(df_facts.loc[index])

                                       break
                                #print('admin ', admin_level, name_admin_level, 'bestmatch ', best_match, score_m, 'edit dist', edit_distance(best_match, name_admin_level), '\n')
                                name_admin_level = best_match
                                n_matches_current_level = sum(df_loc_matches[admin_level + '_name'] ==
                                                      name_admin_level)

                        df_loc_matches = df_loc_matches.loc[
                                df_loc_matches[admin_level + '_name'] == name_admin_level]

                        if (n_matches_current_level) == 0 & (admin_level == level_tag[-1]):
                                n_no_matches +=1 
                        if n_matches_current_level == 1 :

                                n_perfect_matches +=1
                                if verbose : print(df_loc_matches)
                                for admin_level_tag in level_tag: 

                                        df_facts.loc[index, admin_level_tag + '_code'] = (df_loc_matches[admin_level_tag + '_code'].values[0])
                                        df_facts.loc[index, admin_level_tag + '_best_match_name'] = (df_loc_matches[admin_level_tag + '_name'].values[0])
                        upper_level += admin_level + df_facts.loc[index][admin_level + '_name']
                        
                counter+=1

        return df_facts


# In[ ]:


# To be called function
pcode_file(pcode_loc, facts, '../../output/uganda-test.csv', ask_below_score, reject_below_score)


# In[ ]:




