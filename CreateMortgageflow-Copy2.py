#!/usr/bin/env python
# coding: utf-8

# # Reading Mortgage data

# In[7]:


import pandas as pd
import numpy as np


# In[9]:


# Path to the uploaded CSV file
csv_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'
# Read the CSV file with a specified delimiter
mortgages_df = pd.read_csv(csv_path, delimiter=',')
# Set pandas to display all rows
pd.set_option('display.max_rows', None)

print(mortgages_df)


# # variables of unique values from data set Mortgages

# In[3]:


unique_rvps = mortgages_df['Looptijd Rentevastperiode 1'].unique() # indicates the values of fixed interest periods in months 
unique_aflosvorm = mortgages_df['Aflosvorm'].unique() # types of repayment like annuity, linear, etc.
print("Unieke Rentevaste Periodes:", unique_rvps)
print("Unieke Aflosvormen:", unique_aflosvorm)


# # Reading Data from Data set

# In[4]:


num_steps = 360

issue_dt = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel']) # Start date of the loan part
mat_dt = pd.to_datetime(mortgages_df['Datum Eind Leningdeel']) # End date of the loan part
reprice_dt = pd.to_datetime(mortgages_df['Datum Eind RVP']) # End date of the fixed interest period
fixed_per = mortgages_df['Looptijd Rentevastperiode 1'] # Fixed interest period in months
orig_principal = mortgages_df['Hoofdsom Oorspronkelijk'] # Original principal amount
curr_principal = mortgages_df['Hoofdsom Restant'] # Current remaining principal amount
int_rate = mortgages_df['Rente Nominaal'] # Nominal interest rate
red_types = mortgages_df['Aflosvorm'] # Types of repayment
remaining_rvp = mortgages_df['Looptijd Rentevastperiode 1']  # Remaining fixed interest period in months
principal_savings = mortgages_df['Bedrag Banksparen'] + mortgages_df['Bedrag Spaardepot']  # Savings amount for bank savings and savings deposit


#print("issue_dt:", issue_dt)
#print("mat_dt:", mat_dt)
#print("reprice_dt:", reprice_dt)
#print("fixed_per:", fixed_per)
#print("orig_principal:", orig_principal)
#print("curr_principal:", curr_principal)
#print("int_rate:", int_rate)
#print("red_types:", red_types)
#print("remaining_rvp:", remaining_rvp)
#print("principal_savings:", principal_savings)


# #  Creating empty arrays to store data 

# In[5]:


num_morts = len(mortgages_df)
tot_principal = np.zeros((num_morts, num_steps))
tot_redemptions = np.zeros((num_morts, num_steps))
tot_resets = np.zeros((num_morts, num_steps))
tot_interest = np.zeros((num_morts, num_steps))
tot_count = np.zeros((num_morts, num_steps))


# #  Initializations for different mortgage combinations

# In[6]:


mort_types = [''] * num_morts  # Array to store types of mortgages
count_mort_types = np.zeros(num_morts) # Array to count the number of each type of mortgage
tot_fltr = np.zeros(len(mortgages_df), dtype=bool)  # Boolean array to filter mortgages based on conditions


# In[7]:


now_dt = pd.Timestamp('2023-03-31').toordinal()  # Spotdate


# # Calling the MortSched function

# In[8]:


mortgages_df = mortgages_df.sort_values(by=['Aflosvorm', 'Looptijd Rentevastperiode 1']).reset_index(drop=True)

def MortSched(now_dt, issue_dt, mat_dt, reprice_dt, remaining_rvp, red_type, fixed_per, principal_savings, curr_principal, int_rate):
    num_steps = 360
    num_mortgages = len(issue_dt)
    mort_dates = np.zeros((num_mortgages, num_steps))
    interest = np.zeros((num_mortgages, num_steps))
    principal = np.zeros((num_mortgages, num_steps))
    resets = np.zeros((num_mortgages, num_steps))
    mort_count = np.zeros((num_mortgages, num_steps))
    
    for i in range(num_mortgages):
        term = (mat_dt.iloc[i] - issue_dt.iloc[i]).days // 30
        monthly_rate = int_rate.iloc[i] / 12 / 100
        remaining_principal = curr_principal.iloc[i]
        
        for j in range(min(term, num_steps)):
            interest[i, j] = remaining_principal * monthly_rate
            
            if red_type == 'Annuiteit':
                annuity_payment = remaining_principal * (monthly_rate / (1 - (1 + monthly_rate) ** -term))
                principal[i, j] = annuity_payment - interest[i, j]
            elif red_type == 'Linear':
                principal[i, j] = remaining_principal / term
            elif red_type == 'Aflossingsvrij':
                principal[i, j] = 0
            elif red_type == 'Levensverzekering':
                # Simple approach, assuming that life insurance follows the same logic as annuity
                annuity_payment = remaining_principal * (monthly_rate / (1 - (1 + monthly_rate) ** -term))
                principal[i, j] = annuity_payment - interest[i, j]
            else:
                principal[i, j] = 0  # Add other types of repayment if necessary
            
            remaining_principal -= principal[i, j]
            mort_count[i, j] = 1
    
    return mort_dates, interest, principal, resets, mort_count


# # For loop followed by cashflow results

# In[9]:


def create_mortgage_flows():
    for red_idx in range(len(unique_aflosvorm)): #loop through all unique repayment types
        for rvp_idx in range(len(unique_rvps)): #loop through all unique rvps
            aflosvorm = unique_aflosvorm[red_idx]
            rvp = unique_rvps[rvp_idx]
             # Filter for mortgages matching current repayment type and fixed interest period
            fltr = (mortgages_df['Looptijd Rentevastperiode 1'] == rvp) & (mortgages_df['Aflosvorm'] == aflosvorm)
            if fltr.sum() == 0: #skip if no mortgage matches the filter
                continue

            print(f"Processing {aflosvorm} with RVP {rvp}...")
# Process each mortgage that matches the filter
            for idx in mortgages_df[fltr].index:
                mort_dates, interest, principal, resets, mort_count = MortSched(
                    now_dt, 
                    issue_dt[idx:idx+1], 
                    mat_dt[idx:idx+1], 
                    reprice_dt[idx:idx+1], 
                    remaining_rvp[idx:idx+1], 
                    aflosvorm, 
                    fixed_per[idx:idx+1], 
                    principal_savings[idx:idx+1], 
                    curr_principal[idx:idx+1], 
                    int_rate[idx:idx+1]
                )
                
                max_steps = min(fixed_per[idx], 360)  # Limits to RVP's of 360 months
                
                tot_count[idx, :max_steps] = mort_count[0, :max_steps]
                tot_resets[idx, :max_steps] = resets[0, :max_steps]
                tot_redemptions[idx, :max_steps] = -np.diff(np.append([0], np.sum(principal, axis=0)[:max_steps]))
                tot_principal[idx, :max_steps] = np.sum(principal, axis=0)[:max_steps]
                tot_interest[idx, :max_steps] = np.sum(interest[:, :max_steps], axis=0)
                
                mort_types[idx] = f"{aflosvorm}_{rvp}"

                if abs(np.sum(curr_principal[idx:idx+1]) - np.sum(-np.diff(np.sum(principal, axis=0)))) > 1:
                    print(f"Principal mismatch for index {idx}: {abs(np.sum(curr_principal[idx:idx+1]) - np.sum(-np.diff(np.sum(principal, axis=0))))}")

    return tot_principal, tot_redemptions, tot_interest

def print_cashflows(tot_principal, tot_redemptions, tot_interest):
    for i in range(num_morts):
        print(f"Hypotheek {i} met aflosvorm en looptijd {mort_types[i]}:")
        max_steps = min(fixed_per[i], 360)
        for month in range(max_steps):
            print(f"  Cashflow maand {month + 1}: {tot_principal[i, month] + tot_interest[i, month]}")

tot_principal, tot_redemptions, tot_interest = create_mortgage_flows()
print_cashflows(tot_principal, tot_redemptions, tot_interest)


# In[ ]:




