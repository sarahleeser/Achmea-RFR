#!/usr/bin/env python
# coding: utf-8

# In[152]:


import pandas as pd
import numpy as np

# Bestanden inlezen
mortgages_path = '/var/R/home/aa630798/Achmea Mortgagecashflows/Test_mortgages.csv'

# Lees de CSV-bestanden in
mortgages_df = pd.read_csv(mortgages_path, delimiter=',', encoding='ISO-8859-1')

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

# Spotdate instellen
spotdate = pd.Timestamp('2023-03-31')
print(mortgages_df)



# In[153]:


# Unieke waardes bekijken
unique_rvps = mortgages_df['Looptijd Rentevastperiode 1'].unique()
unique_aflosvorm = mortgages_df['Aflosvorm'].unique()
print("Unieke Rentevaste Periodes:", unique_rvps)
print("Unieke Aflosvormen:", unique_aflosvorm)

num_steps = 360
num_morts = len(mortgages_df)

# Bepaal de resterende maanden vanaf de spotdate
mortgages_df['Months After Spotdate'] = (mortgages_df['Datum Eind RVP'].dt.to_period('M') - spotdate.to_period('M')).apply(lambda x: max(0, x.n))

# Pas n_periods aan naar de resterende maanden vanaf de spotdate
mortgages_df['n_periods_spotdate'] = mortgages_df[['Months After Spotdate', 'Looptijd Rentevastperiode 1']].min(axis=1)


# In[154]:


# Bereken het aantal maanden tussen de start- en einddatum zoals in Matlab
def calculate_num_mat_months(start_date, end_date):
    return max((end_date.year - start_date.year) * 12 + (end_date.month - start_date.month), 1)

# Voeg de berekening van num_mat_months toe aan de dataframe
mortgages_df['num_mat_months'] = mortgages_df.apply(lambda row: calculate_num_mat_months(row['Datum Ingang Leningdeel'], row['Datum Eind Leningdeel']), axis=1)

# Gebruik de resterende hoofdsom vanaf de spotdate
mortgages_df['Initial Principal'] = mortgages_df['Hoofdsom Restant']


print("Aantal hypotheken:", num_morts)



# In[155]:


# Mapping van Aflosvormen en Functies voor Cashflow Berekening
# Consistente mapping tussen beide scripts
aflosvorm_mapping = {
    'Annuiteit': 'annuity',
    'Lineair': 'linear',
    'Aflossingsvrij': 'bullet',
    'Levensverzekering': 'savings',  # Behandeld als aflossingsvrij
    'Spaarrekening': 'savings',  # Behandeld als aflossingsvrij
    'Spaarverzekering': 'savings',
    'Beleggingshypotheek': 'savings',
    'Belegging': 'savings',
    'Leven': 'savings',
    'Beleggingsverzekering': 'savings'
}
print("Aflosvorm Mapping:", aflosvorm_mapping)


# In[156]:


# Functie voor annuïteitenberekening
def annuity_cashflows(pv, rate, n_periods_rvp, num_mat_months, months_after_spotdate):
    n_remaining_periods = num_mat_months - months_after_spotdate
    annuity_payment = pv * (rate / (1 - (1 + rate) ** -n_remaining_periods))

    interest = np.zeros(n_periods_rvp)
    principal = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp)

    remaining_balance = pv
    for t in range(n_periods_rvp):
        interest[t] = remaining_balance * rate
        principal[t] = annuity_payment - interest[t]
        remaining_balance -= principal[t]
        balance[t] = remaining_balance

    if remaining_balance > 0:
        principal[-1] += remaining_balance
        balance[-1] = 0

    return interest, principal, balance


# In[157]:


# Functie voor lineaire cashflows
def linear_cashflows(pv, rate, n_periods_rvp, num_mat_months, months_after_spotdate):
    n_remaining_periods = num_mat_months - months_after_spotdate
    monthly_principal_payment = pv / n_remaining_periods

    interest = np.zeros(n_periods_rvp)
    principal = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp)

    remaining_balance = pv
    for t in range(n_periods_rvp):
        interest[t] = remaining_balance * rate
        principal[t] = monthly_principal_payment
        remaining_balance -= principal[t]
        balance[t] = remaining_balance

    if remaining_balance > 0:
        principal[-1] += remaining_balance
        balance[-1] = 0

    return interest, principal, balance


# In[158]:


def interest_only_cashflows(pv, rate, n_periods_rvp, n_periods_total):
    interest = np.full(n_periods_rvp, pv * rate)
    principal = np.zeros(n_periods_rvp)
    balance = np.full(n_periods_rvp, pv)
    
    # Voeg de volledige hoofdsom toe aan de laatste maand
    if n_periods_rvp > 0:
        principal[-1] = pv  # De laatste maand wordt de volledige aflossing van het restant
        balance[-1] = 0  # Na de laatste betaling is het saldo 0

    return interest, principal, balance


# In[159]:


def savings_cashflows(pv, rate, n_periods_rvp, num_mat_months, months_after_spotdate):
    
    # Bereken de resterende periodes
    n_remaining_periods = num_mat_months - months_after_spotdate
    print(f"Resterende periodes: {n_remaining_periods}")  # Printregel toegevoegd

    monthly_rate = rate / 12

    # Bereken de maandelijkse inleg (x) voor de spaarpot (eerste maand)
    monthly_inleg = - (monthly_rate) / ((1 + monthly_rate) ** n_remaining_periods - 1) * -pv

    # Initialiseer arrays voor de principal, interest en balance
    principal = np.zeros(n_periods_rvp)
    interest = np.zeros(n_periods_rvp)
    balance = np.zeros(n_periods_rvp + 1)  # We houden één extra plek voor het beginsaldo
    savings_pot = np.zeros(n_periods_rvp + 1)  # +1 om de beginwaarde op 0 te houden

    # Zet de eerste maand
    savings_pot[0] = 0  # Initieel saldo
    balance[0] = pv  # Initieel beginsaldo hypotheek

    # Bereken de aflossingen, interest en balans voor de maanden
    for t in range(n_periods_rvp):
        # Spaarpot groeit door inleg en rente
        savings_pot[t + 1] = savings_pot[t] * (1 + monthly_rate) + monthly_inleg
        # Aflossing is het verschil tussen de huidige en vorige maand
        principal[t] = savings_pot[t + 1] - savings_pot[t]
        # Rente op het uitstaande saldo (de hypotheek die nog niet is afgelost)
        interest[t] = balance[t] * monthly_rate
        # Resterend saldo na aflossing
        balance[t + 1] = pv - savings_pot[t + 1]

    # Controleer of er een resterend saldo is aan het eind en corrigeer dit
    if balance[-1] > 0:
        principal[-1] += balance[-1]
        balance[-1] = 0

    return interest, principal, balance[:-1]  # balance[:-1] om de laatste waarde niet mee te nemen


# In[162]:


def calculate_cashflows(mortgages_df, spotdate):
    cashflows = []
    num_steps = 360

    for idx, row in mortgages_df.iterrows():
        aflosvorm = aflosvorm_mapping.get(row['Aflosvorm'], 'unknown')
        n_periods_rvp = row['n_periods_spotdate']
        num_mat_months = row['num_mat_months']
        months_after_spotdate = (spotdate.to_period('M') - row['Datum Ingang Leningdeel'].to_period('M')).n

        if aflosvorm == 'annuity':
            interest, principal, balance = annuity_cashflows(
                row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, num_mat_months, months_after_spotdate
            )
        elif aflosvorm == 'linear':
            interest, principal, balance = linear_cashflows(
                row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, num_mat_months, months_after_spotdate
            )
        elif aflosvorm == 'bullet' or aflosvorm == 'Aflossingsvrij':
            interest, principal, balance = interest_only_cashflows(
                row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, num_mat_months
            )
        elif aflosvorm == 'savings' or aflosvorm == 'Levensverzekering':
            # Voeg het ontbrekende argument months_after_spotdate toe
            interest, principal, balance = savings_cashflows(
                row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, num_mat_months, months_after_spotdate
            )
        elif aflosvorm == 'investment':
            interest, principal, balance = investment_cashflows(
                row['Initial Principal'], row['Rente Nominaal'] / 12, n_periods_rvp, num_mat_months
            )
        else:
            raise ValueError(f"Onbekende aflosvorm: {row['Aflosvorm']}")

        # Vul de resterende periodes met nullen (om te zorgen dat alle arrays 360 lang zijn)
        interest = np.pad(interest, (0, num_steps - len(interest)), 'constant')
        principal = np.pad(principal, (0, num_steps - len(principal)), 'constant')
        balance = np.pad(balance, (0, num_steps - len(balance)), 'constant')

        cashflows.append({
            'HypotheekID': row['Leningdeel'],
            'Aflosvorm': aflosvorm,
            'Interest': interest,
            'Principal': principal,
            'Balance': balance
        })

    return cashflows


# In[163]:


# Cashflows berekenen
cashflows = calculate_cashflows(mortgages_df, spotdate)

# Print de cashflows voor alle hypotheken
for idx, cashflow in enumerate(cashflows):
    print(f"Hypotheek {idx + 1}:")
    print(f"Hypotheek ID: {cashflow['HypotheekID']}")
    print(f"Aflosvorm: {cashflow['Aflosvorm']}")
    print("Interest Cashflows:", cashflow['Interest'])
    print("Principal Cashflows:", cashflow['Principal'])
    print("Balance:", cashflow['Balance'])
    print("\n")


# In[104]:


# Zet de cashflows om naar een DataFrame
def cashflows_to_dataframe(cashflows):
    data = []
    for cashflow in cashflows:
        for month in range(len(cashflow['Interest'])):
            data.append({
                'HypotheekID': cashflow['HypotheekID'],
                'Aflosvorm': cashflow['Aflosvorm'],
                'Month': month + 1,
                'Interest': cashflow['Interest'][month],
                'Principal': cashflow['Principal'][month],
                'Balance': cashflow['Balance'][month]
            })
    return pd.DataFrame(data)

# Zet de cashflows om naar een DataFrame
cashflows_df = cashflows_to_dataframe(cashflows)

# Sla de DataFrame op als CSV-bestand
cashflows_df.to_csv('cashflows_without_survival.csv', index=False)

print("Cashflows zijn opgeslagen in 'cashflows_without_survival.csv'")


# In[ ]:





# In[ ]:





# In[ ]:




