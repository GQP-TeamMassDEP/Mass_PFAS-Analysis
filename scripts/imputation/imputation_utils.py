import numpy as np
import pandas as pd
import geocoder
import numpy as np

def geocode(addresses, address_col):
    
    lons = []
    lats = []
    for address in addresses[address_col]:

        g = geocoder.osm(address)

        try:
            lons.append(g.osm['x'])
            lats.append(g.osm['y'])

        except: 
            print(f'{address} : No results found')
            lons.append(np.nan)
            lats.append(np.nan)
            
    return lats, lons

def remove_blanks(df, matrix_col, report_col, address_col, sample_id_col):
    
    # If matrix is non-existant put 'Liquid' if AFFF, otherwise Water. This can happen for some compounds within a sample
    if df[matrix_col].isna().sum() > 0:
        df[matrix_col] = np.where((df[matrix_col].isna()), 'water', df[matrix_col])
        df[matrix_col] = np.where((df[matrix_col].isna()) & (df[report_col].str.lower().str.contains('afff')), 'liquid', df[matrix_col])
        
    # If sample_id is non-existant just put other. This only happens for an entire sample_id
    if df[sample_id_col].isna().sum() > 0:
        df[sample_id_col] = df[sample_id_col].replace({np.nan : 'other'})
    
    # Remove all blanks from dataset
    df = df[~(df[matrix_col].str.lower().str.contains('blank'))]
    df = df[~(df[sample_id_col].str.lower().str.contains('blank'))]
    df = df[~(df[sample_id_col].str.lower().str.contains('fb'))]
    df = df[~(df[sample_id_col].str.lower().str.contains('eff'))]
    df = df[~(df[sample_id_col].str.lower().str.contains('mid'))]
    df = df[~(df[sample_id_col].str.lower().str.contains('treat'))]
    df = df[~(df[address_col].astype(str).str.lower().str.contains('trip blank'))]
    df = df[~(df[address_col].astype(str).str.lower().str.contains('tb'))]
    df = df[~(df[address_col].astype(str).str.lower().str.contains(' mid'))]
    df = df[~(df[address_col].astype(str).str.lower().str.contains(' eff'))]
    
    
    return df

def convert_units(df, units_col, matrix_col, measurement_cols):
    
    # Not sure what to do with '(cid:181)g/kg dry'
    
    df[units_col] = df[units_col].str.lower()

    for col in measurement_cols:
        
        df[col] = np.where(df[units_col] == 'ug/l', (df[col] * 1000), df[col])
        df[col] = np.where(df[units_col] == '(cid:181)g/kg dry', (df[col] * 1000), df[col])
        
    df[units_col] = df[units_col].replace({'ug/l' : 'ng/l',
                                           'ng/i' : 'ng/l',
                                           'ng/1' : 'ng/l',
                                           'j' : 'ng/l',
                                           'ug/kg' : 'ng/g',
                                           '(cid:181)g/kg dry' : 'ng/g'})
    
    # Fill in NA units with either ng/l or g/kg (water or soil)
    df[units_col] = np.where(df[units_col].isna() & 
                   (df[matrix_col].str.lower().str.contains('water') | df[matrix_col].str.lower().str.contains('dw')),
                    'ng/l',
                   df[units_col])
    
    df[units_col] = np.where(df[units_col].isna() & 
                   (df[matrix_col].str.lower().str.contains('so')),
                    'ng/g',
                   df[units_col])
    
    return df

def check_RL(df, rl_col, output_col):
    
    if df[df[rl_col].isna()][output_col].isna().sum() == 0:
        print('No RL issues. All good for imputation!')
        
#     else: # Unnecessary as isn't a current issue

    return df

def create_censored_col(df, result_col, output_col, limit_col):
    
    df[result_col] = pd.to_numeric(df[result_col], errors = 'coerce')
    
    # Prepare data for ROS by creating a column showing where it is censored and imputing the limit in those instances
    df[f'{output_col}_cen'] = np.where(df[result_col].isna(), 1, 0)
    df[output_col] = np.where(df[f'{output_col}_cen'] == 1, df[limit_col], df[result_col])
    
    return df

def dilute_measurements(df, dilution_factor_col, measurement_cols):

    # can assume if no df --> 1
    df[dilution_factor_col] = df[dilution_factor_col].replace({np.nan : 1})
    
    for col in measurement_cols:
        
        df[col] = df[col] / df[dilution_factor_col]
        
    return df

def fill_na_with_mdl_rl(df, pfas_vars, rl_mdl_lookup):

    for row in df.iterrows():

        # If there are NA compounds
        if row[1].isna().sum() > 0:

            # get lab
            lab = row[1]['lab']

            # for each compound that exists in testing
            for key in pfas_vars:
                
                if key in row[1].keys():
                    # If it is na - replace with rl_mdl_lookup
                    if (np.isnan(row[1][key])):
                        x = rl_mdl_lookup[(rl_mdl_lookup['lab'] == lab) & (rl_mdl_lookup['Acronym'] == key)][['RL', 'MDL']].values.min()
                        df.loc[row[0], key] = x / 2
                        
    return df

def create_mrl_mdl_dict(residential_df, pfas18):
    mrl_dict = {}
    mdl_dict = {}
    for var in pfas18:
        # Only keep those that are available
        if var in residential_df.columns:
            mrl_dict[var] = residential_df.columns[residential_df.columns.get_loc(var) + 1]
            mdl_dict[var] = residential_df.columns[residential_df.columns.get_loc(var) + 2]
            
    pfas_vars = list(mrl_dict.keys())
    
    return mrl_dict, mdl_dict, pfas_vars

def create_max_disposal_df(disposal_source_df_wide): 
    ns_disposal_source_df = pd.DataFrame()
    s_disposal_source_df = pd.DataFrame()
    for rtn in disposal_source_df_wide['RTN'].unique():
        
        # Find max by soil v. water - if water exists ignore soil
        soil_data = disposal_source_df_wide[(disposal_source_df_wide['Matrix'].str.lower().str.contains('so'))]
        non_soil_data = disposal_source_df_wide[~(disposal_source_df_wide['Matrix'].str.lower().str.contains('so'))]

        # filter down dataset & # for each pfas_col find the max
        s_max_row = pd.DataFrame(soil_data[soil_data['RTN'] == rtn].max(axis = 0)).T
        ns_max_row = pd.DataFrame(non_soil_data[non_soil_data['RTN'] == rtn].max(axis = 0)).T

        ns_disposal_source_df = pd.concat([ns_disposal_source_df,ns_max_row],axis=0)
        s_disposal_source_df = pd.concat([s_disposal_source_df,s_max_row],axis=0)
        
    # drop na RTNs
    ns_disposal_source_df = ns_disposal_source_df[~(ns_disposal_source_df['RTN'].isna())]

    # Find RTNs in soil not in ns
    only_soil_rtns = list(set(s_disposal_source_df['RTN']) - set(ns_disposal_source_df['RTN']))
    only_soil_rtns_df = s_disposal_source_df[s_disposal_source_df['RTN'].isin(only_soil_rtns)]
    only_soil_rtns_df = only_soil_rtns_df[~(only_soil_rtns_df['RTN'].isna())]

    # Concatenate those rows directly
    max_disposal_source_df = pd.concat([ns_disposal_source_df, only_soil_rtns_df], axis = 0)
    max_disposal_source_df = max_disposal_source_df.reset_index()
    # Find all nan values in s, and replace with ns (if both na - stays na)

    s_disposal_source_df = s_disposal_source_df[~(s_disposal_source_df['RTN'].isna())]
    soil_with_rtns = list(s_disposal_source_df['RTN'].unique())

    s_disposal_source_df  = s_disposal_source_df.set_index('RTN')
    max_disposal_source_df = max_disposal_source_df.set_index('RTN')
    
    max_disposal_source_df = max_disposal_source_df.reset_index()

return max_disposal_source_df