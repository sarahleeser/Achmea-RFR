#!/usr/bin/env python
# coding: utf-8

# # Inlezen data mortages 

# In[80]:


import pandas as pd
import numpy as np


# In[83]:


# Pad naar het geüploade CSV-bestand
csv_path = '/var/R/home/aa630798/Test_mortgages.csv'
# Lees het CSV-bestand in met een gespecificeerde delimiter
mortgages_df = pd.read_csv(csv_path, delimiter=',')
# Stel pandas in om alle rijen weer te geven
pd.set_option('display.max_rows', None)
# Bekijk de DataFrame
#print(mortgages_df)


# # variabelen van unieke waarde uitgelezen uit mortgages
# 

# In[84]:


uniqueRvps = mortgages_df['Looptijd Rentevastperiode 1'].unique() #geeft aan welke waarde van rentevaste periode er zijn in maanden
uniqueAflosvorm = mortgages_df['Aflosvorm'].unique() #soorten aflossen zoals annuiteit, linear etc.
print("Unieke Rentevaste Periodes:", unique_rvps)
print("Unieke Aflosvormen:", unique_aflosvorm)


# In[85]:


# num_morts = len(unique_rvps) * len(unique_aflosvorm) #aantal verschillende combinaties van aflosvormen en rentevaste perioden dus soorten hypotheken
# num_steps = 360
# print("Combinaties soorten mortgages:", num_morts )


# # Functie aanroepen
# 

# # Maken array met begin waarde van elke hoofdsom

# In[86]:


# Lijst om de initiële hoofdsommen op te slaan
#initial_principals = []


# # Maken van lege arrays om later gegevens in op te slaan

# In[93]:


numMorts = len(mortgages_df)
numSteps = 360
mortDates = np.zeros((1, numSteps))

totPrincipal = np.zeros((numMorts, numSteps + 1))
totRedemptions = np.zeros((numMorts, numSteps))
totResets = np.zeros((numMorts, numSteps))
totInterest = np.zeros((numMorts, numSteps))
totCount = np.zeros((numMorts, numSteps))
print("Nummorts:", numMorts)


# In[96]:


# Extract relevant columns from the DataFrame
mortgages_df['Datum Ingang Leningdeel'] = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel'])  # Deze regel is aangepast
mortgages_df['Datum Eind Leningdeel'] = pd.to_datetime(mortgages_df['Datum Eind Leningdeel'])  # Deze regel is aangepast
mortgages_df['Datum Eind RVP'] = pd.to_datetime(mortgages_df['Datum Eind RVP'])  # Deze regel is aangepast

issueDt = mortgages_df['Datum Ingang Leningdeel']
matDt = mortgages_df['Datum Eind Leningdeel'] 
repriceDt = mortgages_df['Datum Eind RVP']
fixedPer = mortgages_df['Looptijd Rentevastperiode 1']
origPrincipal = mortgages_df['Hoofdsom Oorspronkelijk']
currPrincipal = mortgages_df['Hoofdsom Restant']
intRate = mortgages_df['Rente Nominaal']
redTypes = mortgages_df['Aflosvorm']
remainingRVP = mortgages_df['Looptijd Rentevastperiode 1']
PrincipalSavings = mortgages_df['Bedrag Banksparen'] + mortgages_df['Bedrag Spaardepot']

# Display the extracted columns
#print("issueDt:", issueDt)
#print("matDt:", matDt)
#print("repriceDt:", repriceDt)
#print("fixedPer:", fixedPer)
#print("origPrincipal:", origPrincipal)
#print("currPrincipal:", currPrincipal)
#print("intRate:", intRate)
#print("redTypes:", redTypes)
#print("remainingRVP:", remainingRVP)
#print("PrincipalSavings:", PrincipalSavings)


# # initialisaties voor basis verschillende hypothekeek combinaties

# In[97]:


mortTypes = [''] * numMorts
countMortTypes = np.zeros(numMorts)
totFltr = np.zeros(len(mortgages_df), dtype=bool)
nowDt = pd.Timestamp('2023-03-31').toordinal()  # Spotdate
mortgages_df = mortgages_df.sort_values(by=['Aflosvorm', 'Looptijd Rentevastperiode 1']).reset_index(drop=True)


# In[98]:


def MortSched(nowDt, issueDt, matDt, repriceDt, remainingRVP, redType, fixedPer, PrincipalSavings, currPrincipal, intRate):
    numSteps = 360
    numMortgages = len(issueDt)
    mortDates = np.zeros((numMortgages, numSteps))
    interest = np.zeros((numMortgages, numSteps))
    principal = np.zeros((numMortgages, numSteps))
    resets = np.zeros((numMortgages, numSteps))
    mortCount = np.zeros((numMortgages, numSteps))
    
    for i in range(numMortgages):
        term = (matDt.iloc[i] - issueDt.iloc[i]).days // 30
        monthly_rate = intRate.iloc[i] / 12 / 100
        remaining_principal = currPrincipal.iloc[i]
        
        for j in range(min(term, numSteps)):
            interest[i, j] = remaining_principal * monthly_rate
            
            if redType == 'Annuiteit':
                annuity_payment = remaining_principal * (monthly_rate / (1 - (1 + monthly_rate) ** -term))
                principal[i, j] = annuity_payment - interest[i, j]
            elif redType == 'Linear':
                principal[i, j] = remaining_principal / term
            elif redType == 'Aflossingsvrij':
                principal[i, j] = 0
            elif redType == 'Levensverzekering':
                # Simpele benadering, aanname dat levensverzekering dezelfde logica volgt als annuiteit
                annuity_payment = remaining_principal * (monthly_rate / (1 - (1 + monthly_rate) ** -term))
                principal[i, j] = annuity_payment - interest[i, j]
            else:
                principal[i, j] = 0  # Voeg hier andere aflosvormen toe indien nodig
            
            remaining_principal -= principal[i, j]
            mortCount[i, j] = 1
    
    return mortDates, interest, principal, resets, mortCount


# In[99]:


#def MortSched(nowDt, issueDt, matDt, repriceDt, remainingRVP, redType, fixedPer, PrincipalSavings, currPrincipal, intRate):
    # Placeholder function, must be translated from MATLAB to Python
    #numSteps = 360
    #numMortgages = len(issueDt)
    #mortDates = np.zeros((numSteps,))
    #interest = np.zeros((numMortgages, numSteps))
    #principal = np.zeros((numMortgages, numSteps))
    #resets = np.zeros((numMortgages, numSteps))
    #mortCount = np.zeros((numMortgages, numSteps))
    #return mortDates, interest, principal, resets, mortCount


# In[100]:


def create_mortgage_flows():
    for idx in range(len(mortgages_df)):
        fltr = mortgages_df.index == idx
        redType = mortgages_df.loc[fltr, 'Aflosvorm'].values[0]
        rvp = mortgages_df.loc[fltr, 'Looptijd Rentevastperiode 1'].values[0]
        
        mortTypes[idx] = f"{redType}_{rvp}"
        countMortTypes[idx] = 1  # Each entry is a unique mortgage

        mortDates, interest, principal, resets, mortCount = MortSched(
            nowDt, 
            issueDt[fltr], 
            matDt[fltr], 
            repriceDt[fltr], 
            remainingRVP[fltr], 
            redType, 
            fixedPer[fltr], 
            PrincipalSavings[fltr], 
            currPrincipal[fltr], 
            intRate[fltr]
        )
        
        totCount[idx, :] = mortCount.mean(axis=0)
        totResets[idx, :] = resets.mean(axis=0)
        totRedemptions[idx, :-1] = -np.diff(np.sum(principal, axis=0))  # Adjusting the length
        totPrincipal[idx, :-1] = np.sum(principal, axis=0)
        totInterest[idx, :-1] = np.sum(interest[:, :-1], axis=0)
        
        if abs(np.sum(currPrincipal[fltr]) - np.sum(-np.diff(np.sum(principal, axis=0)))) > 1:
            print(abs(np.sum(currPrincipal[fltr]) - np.sum(-np.diff(np.sum(principal, axis=0)))))

    return totPrincipal, totRedemptions, totInterest

def print_cashflows(totPrincipal, totRedemptions, totInterest):
    for i in range(numMorts):
        print(f"Hypotheek {i} met aflosvorm {mortTypes[i]}:")
        for month in range(numSteps):
            print(f"  Cashflow maand {month + 1}: {totPrincipal[i, month] + totInterest[i, month]}")

totPrincipal, totRedemptions, totInterest = create_mortgage_flows()
print_cashflows(totPrincipal, totRedemptions, totInterest)


# In[ ]:




