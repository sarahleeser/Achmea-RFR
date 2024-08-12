#!/usr/bin/env python
# coding: utf-8

# In[46]:


import pandas as pd
import numpy as np

# Bestanden inlezen
mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'


# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')


# Voeg prepayment type toe aan hypotheken data
mortgages_df = mortgages_df.merge(mapping_df, left_on='Aflosvorm', right_on='AflossingsvormOmschrijving', how='left')

# Datum parsing functie
def parse_dates(date_str):
    if isinstance(date_str, pd.Timestamp):
        return date_str
    for fmt in ('%m/%d/%Y', '%m-%d-%Y', '%Y-%m-%d', '%d-%m-%Y'):
        try:
            return pd.to_datetime(date_str, format=fmt)
        except (ValueError, TypeError):
            continue
    return pd.NaT

# Pas de aangepaste datumparser toe
mortgages_df['Datum Ingang Leningdeel'] = mortgages_df['Datum Ingang Leningdeel'].apply(parse_dates)
mortgages_df['Datum Eind RVP'] = mortgages_df['Datum Eind RVP'].apply(parse_dates)
mortgages_df['Datum Eind Leningdeel'] = mortgages_df['Datum Eind Leningdeel'].apply(parse_dates)

print(mortgages_df)


# In[47]:


unique_rvps = mortgages_df['Looptijd Rentevastperiode 1'].unique()
unique_aflosvorm = mortgages_df['Aflosvorm'].unique()
print("Unieke Rentevaste Periodes:", unique_rvps)
print("Unieke Aflosvormen:", unique_aflosvorm)




# In[48]:


num_steps = 360
num_morts = len(mortgages_df)

issue_dt = pd.to_datetime(mortgages_df['Datum Ingang Leningdeel'])
mat_dt = pd.to_datetime(mortgages_df['Datum Eind Leningdeel'])
reprice_dt = pd.to_datetime(mortgages_df['Datum Eind RVP'])
fixed_per = mortgages_df['Looptijd Rentevastperiode 1']
orig_principal = mortgages_df['Hoofdsom Oorspronkelijk']
curr_principal = mortgages_df['Hoofdsom Restant']
int_rate = mortgages_df['Rente Nominaal']
red_types = mortgages_df['Aflosvorm']
remaining_rvp = mortgages_df['Looptijd Rentevastperiode 1']
principal_savings = mortgages_df['Bedrag Banksparen'] + mortgages_df['Bedrag Spaardepot']

# Initieer arrays voor de resultaten
tot_principal = np.zeros((num_morts, num_steps))
tot_redemptions = np.zeros((num_morts, num_steps))
tot_resets = np.zeros((num_morts, num_steps))
tot_interest = np.zeros((num_morts, num_steps))
tot_count = np.zeros((num_morts, num_steps))
mort_types = [''] * num_morts
count_mort_types = np.zeros(num_morts)
tot_fltr = np.zeros(len(mortgages_df), dtype=bool)
now_dt = pd.Timestamp('2023-03-31').toordinal()

# Controleer variabelen en initialisaties
print("Aantal hypotheken:", num_morts)


# In[49]:


# Consistente mapping tussen beide scripts
aflosvorm_mapping = {
    'Annuiteit': 'annuity',
    'Lineair': 'linear',
    'Aflossingsvrij': 'bullet',
    'Levensverzekering': 'bullet',  # Behandeld als aflossingsvrij
    'Spaarrekening': 'bullet',  # Behandeld als aflossingsvrij
    'Spaarverzekering': 'savings',
    'Beleggingshypotheek': 'investment',
    'Belegging': 'investment',
    'Leven': 'savings',
    'Beleggingsverzekering': 'investment'
}
print("Aflosvorm Mapping:", aflosvorm_mapping)


# In[50]:


def annuity_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.zeros(n_periods)
    balance = np.zeros(n_periods)
    annuity_payment = pv * (rate / (1 - (1 + rate) ** -n_periods))
    remaining_balance = pv

    for t in range(n_periods):
        interest[t] = remaining_balance * rate
        principal[t] = annuity_payment - interest[t]
        remaining_balance -= principal[t]
        balance[t] = remaining_balance

    return interest, principal, balance


# In[51]:


def linear_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.full(n_periods, pv / n_periods)
    balance = np.zeros(n_periods)
    remaining_balance = pv

    for t in range(n_periods):
        interest[t] = remaining_balance * rate
        remaining_balance -= principal[t]
        balance[t] = remaining_balance

    return interest, principal, balance


# In[52]:


def interest_only_cashflows(pv, rate, n_periods):
    interest = np.full(n_periods, pv * rate)
    principal = np.zeros(n_periods)
    balance = np.full(n_periods, pv)
    return interest, principal, balance



# In[53]:


def savings_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.zeros(n_periods)
    balance = np.zeros(n_periods)
    # Logic for savings cashflows can be added here
    return interest, principal, balance

def investment_cashflows(pv, rate, n_periods):
    interest = np.zeros(n_periods)
    principal = np.zeros(n_periods)
    balance = np.zeros(n_periods)
    # Logic for investment cashflows can be added here
    return interest, principal, balance


# In[54]:


def calculate_cashflows(mortgages_df):
    cashflows = []
    num_steps = 360

    for idx, row in mortgages_df.iterrows():
        aflosvorm = aflosvorm_mapping.get(row['Aflosvorm'], 'unknown')
        n_periods = min(row['Looptijd Rentevastperiode 1'], num_steps)
        
        if aflosvorm == 'annuity':
            interest, principal, balance = annuity_cashflows(row['Hoofdsom Restant'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'linear':
            interest, principal, balance = linear_cashflows(row['Hoofdsom Restant'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'bullet' or aflosvorm == 'Aflossingsvrij':
            interest, principal, balance = interest_only_cashflows(row['Hoofdsom Restant'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'savings' or aflosvorm == 'Levensverzekering':
            interest, principal, balance = savings_cashflows(row['Hoofdsom Restant'], row['Rente Nominaal'] / 12, n_periods)
        elif aflosvorm == 'investment':
            interest, principal, balance = investment_cashflows(row['Hoofdsom Restant'], row['Rente Nominaal'] / 12, n_periods)
        else:
            raise ValueError(f"Onbekende aflosvorm: {row['Aflosvorm']}")

        # Vul de resterende periodes met nullen
        interest = np.pad(interest, (0, num_steps - n_periods), 'constant')
        principal = np.pad(principal, (0, num_steps - n_periods), 'constant')
        balance = np.pad(balance, (0, num_steps - n_periods), 'constant')

        cashflows.append({
            'HypotheekID': row['Leningdeel'],
            'Aflosvorm': aflosvorm,
            'Interest': interest,
            'Principal': principal,
            'Balance': balance
        })

    return cashflows



# In[55]:


# Cashflows berekenen
cashflows = calculate_cashflows(mortgages_df)

# Print de cashflows voor alle hypotheken
for idx, cashflow in enumerate(cashflows):
    print(f"Hypotheek {idx + 1}:")
    print(f"Hypotheek ID: {cashflow['HypotheekID']}")
    print(f"Aflosvorm: {cashflow['Aflosvorm']}")
    print("Interest Cashflows:", cashflow['Interest'])
    print("Principal Cashflows:", cashflow['Principal'])
    print("Balance:", cashflow['Balance'])
    print("\n")


# In[ ]:





# In[ ]:




