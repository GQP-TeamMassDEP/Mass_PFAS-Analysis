from numpy import eye, asarray, dot, sum, diag
from numpy.linalg import svd
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn import preprocessing
import random
import geopandas as gpd
import copy
from sklearn.decomposition import PCA
from sklearn.utils import resample
from sklearn.linear_model import LinearRegression, PoissonRegressor
from sklearn.metrics import mean_squared_error
import statsmodels.api as sm
from scipy import stats
import numpy as np, scipy.stats as st

### Data Cleaning & Imputation ####

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

def find_values_less_than_num(pfas_vars, residential_df):
    # find all '<#' like <1.1 values in pfas 18
    less_than_number = []
    for pfas in pfas_vars:
        strings = residential_df[pfas].astype(str).unique()
        less_than_number.append([string for string in strings if '<' in string])

    flat_list = set([item for sublist in less_than_number for item in sublist])
    flat_list.remove('<MRL')
    
    return flat_list

def impute_na_with_half_mrl_mdl(mrl_dict, mdl_dict, residential_df):
    
    metadict = {}
    for pfas in mrl_dict.keys():
        metadict[pfas] = {}
        # Turn all non-numeric into np.nan for each pfas, mdl and mrl column
        residential_df[pfas] = pd.to_numeric(residential_df[pfas], errors = 'coerce') 
        residential_df[mrl_dict[pfas]] = pd.to_numeric(residential_df[mrl_dict[pfas]], errors = 'coerce')
        residential_df[mdl_dict[pfas]] = pd.to_numeric(residential_df[mdl_dict[pfas]], errors = 'coerce')

        # find the minimum between the two columns
        residential_df['min_mrl_mdl'] = residential_df[[mrl_dict[pfas], mdl_dict[pfas]]].min(axis=1)
        
        # replace pfas sampling result nas with min
        residential_df[pfas] = np.where(residential_df[pfas].isna(), residential_df[mrl_dict[pfas]] / 2, residential_df[pfas])

        ## Checks
        # Check to make sure that if MDL exists <-- it is lower
        metadict[pfas]['MDL < MRL'] = any(residential_df[~(residential_df[mdl_dict[pfas]].isna())][mdl_dict[pfas]] == residential_df[~(residential_df[mdl_dict[pfas]].isna())]['min_mrl_mdl'])

        # Check to see if any nans remaning
        metadict[pfas]['% NAs'] = round(residential_df['min_mrl_mdl'].isna().sum() / residential_df.shape[0] * 100, 2)

    return residential_df, metadict

def remove_blanks(df, matrix_col, sample_id_col):
    
    # If matrix is non-existant just put other
    if df[matrix_col].isna().sum() > 0:
        df[matrix_col] = df[matrix_col].replace({np.nan : 'other'})
        
    # If sample_id is non-existant just put other
    if df[sample_id_col].isna().sum() > 0:
        df[sample_id_col] = df[sample_id_col].replace({np.nan : 'other'})
    
    # Remove all blanks from dataset
    df = df[~(df[matrix_col].str.lower().str.contains('blank'))]
    df = df[~(df[sample_id_col].str.lower().str.contains('blank'))]
    
    
    return df

def get_pfas_vars(df, pfas18):
    pfas_dict = {}
    for var in pfas18:
        if var in df.columns:
            pfas_dict[var] = df.columns[df.columns.get_loc(var)]

    pfas_vars = list(pfas_dict.keys())
    
    return pfas_vars

#### Normalization ####
def partition_by_UoM(df, units_col):
    
    #create unique list of units
    unique_units = df[units_col].str.lower().unique()

    #create a data frame dictionary to store your data frames
    df_dict = {elem : pd.DataFrame for elem in unique_units}

    for key in df_dict.keys():
        df_dict[key] = df[:][df[units_col].str.lower() == key]
        df_dict[key] = df_dict[key].reset_index(drop = True)
    
    return df_dict

def normalize(df, pfas_vars, scaler_fn):
    x = df[pfas_vars].values #returns a numpy array
    
    if scaler_fn == 'MinMax':
        scaler_output = preprocessing.MinMaxScaler()
    
    if scaler_fn == 'Zscore':
        scaler_output = preprocessing.StandardScaler()
    
    x_scaled = scaler_output.fit_transform(x)
    df[pfas_vars] = pd.DataFrame(x_scaled)
    
    return df

def normalize_over_partitions(df, pfas_vars, units_col, scaler_fn):
    
    df_dict = partition_by_UoM(df, units_col = units_col)
    
    df_list = []
    for k,v in df_dict.items():
        
        normalized_df = normalize(df = v.copy(), 
                                  pfas_vars = pfas_vars,
                                  scaler_fn = scaler_fn)

        df_list.append(normalized_df)
    
    full_normalized_df = pd.concat(df_list)
    
    return full_normalized_df
    
#### PCA functions and plotting #####
def varimax(Phi, gamma = 1, q = 20, tol = 1e-6):
    p,k = Phi.shape
    R = eye(k)
    d=0
    for i in range(q):
        d_old = d
        Lambda = dot(Phi, R)
        u,s,vh = svd(dot(Phi.T,asarray(Lambda)**3 - (gamma/p) * dot(Lambda, diag(diag(dot(Lambda.T,Lambda))))))
        R = dot(u,vh)
        d = sum(s)
        if d/d_old < tol: break
    return dot(Phi, R)


def pcaplot(x=None, y=None, z=None, labels=None, var1=None, var2=None, var3=None, axlabelfontsize=9,
                axlabelfontname="Arial", figtype='png', r=300, show=False, plotlabels=True, dim=(10, 8), theme=None):
    if x is not None and y is not None and z is None:
        assert var1 is not None and var2 is not None and labels is not None, "var1 or var2 variable or labels are missing"
        plt.subplots(figsize=dim)
        for i, varnames in enumerate(labels):
            plt.scatter(x[i], y[i])
            plt.text(x[i], y[i], varnames, fontsize=10)
        plt.xlabel("PC1 ({}%)".format(var1))
        plt.ylabel("PC2 ({}%)".format(var2))
        
    elif x is not None and y is not None and z is not None:
        assert var1 and var2 and var3 and labels is not None, "var1 or var2 or var3 or labels are missing"
        # for 3d plot
        fig = plt.figure(figsize=dim)
        ax = fig.add_subplot(111, projection='3d')
        for i, varnames in enumerate(labels):
            ax.scatter(x[i], y[i], z[i])
            if plotlabels:
                ax.text(x[i], y[i], z[i], varnames, fontsize=10)
        ax.set_xlabel("PC1 ({}%)".format(var1), fontsize=axlabelfontsize, fontname=axlabelfontname)
        ax.set_ylabel("PC2 ({}%)".format(var2), fontsize=axlabelfontsize, fontname=axlabelfontname)
        ax.set_zlabel("PC3 ({}%)".format(var3), fontsize=axlabelfontsize, fontname=axlabelfontname)
#         general.get_figure(show, r, figtype, 'pcaplot_3d', theme)
        
def pca_loadings_plot(score, coeff, labels=None, title = None):
    xs = score[:,0]
    ys = score[:,1]
    n = coeff.shape[0]
    scalex = 1.0/(xs.max() - xs.min())
    scaley = 1.0/(ys.max() - ys.min())
    plt.figure(figsize=(14, 10), dpi=80)
    plt.scatter(xs * scalex,ys * scaley,s=5)
    for i in range(n):
        plt.arrow(0, 0, coeff[i,0], coeff[i,1],color = 'r',alpha = 0.5)
        if labels is None:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, "Var"+str(i+1), color = 'green', ha = 'center', va = 'center')
        else:
            plt.text(coeff[i,0]* 1.15, coeff[i,1] * 1.15, labels[i], color = 'g', ha = 'center', va = 'center')
 
    plt.xlabel("PC{}".format(1))
    plt.ylabel("PC{}".format(2))
    plt.title(title)
    plt.grid()
    
### Spatial Inference ####
def extract_rtn(df, report_col):
    '''
    Specific for source and receptor reports (since they start with the RTN)
    '''
    
    rtn_col = df[report_col].str.split('-').str[:2]
    
    cleaned_rtns = []
    for rtn in rtn_col:
        a = '-'.join(rtn)
        a = a.strip()
        cleaned_rtns.append(a)
        
    return cleaned_rtns

def add_wiggliness_for_spatial_interpolation(df, lat_col, lon_col):
    '''
    In order to better use the map, we add a little random wiggle to the RTN lat/lons, so they are differentiable on the map, and not just on top of one another.
    '''
    
    rand_wiggly_lat = np.random.uniform(-0.01, 0.01, df.shape[0])
    rand_wiggly_lon = np.random.uniform(-0.01, 0.01, df.shape[0])
    
    df[lat_col] = df[lat_col] + rand_wiggly_lat
    df[lon_col] = df[lon_col] + rand_wiggly_lon
    
    return df

## Attribution Modeling ##
def create_APCS(pca, scaled_x):
    
    rawLoadings = np.dot(pca.components_.T, np.diag(np.sqrt(pca.explained_variance_)))
    rotated_loadings = varimax(rawLoadings)
    invLoadings = np.linalg.pinv(rotated_loadings.T)
    scores = np.dot(preprocessing.StandardScaler().fit_transform(scaled_x), invLoadings)
    z0i = -np.mean(scaled_x, axis = 0) / np.sqrt(np.var(scaled_x))
    scores0 = np.dot(z0i, invLoadings)
    scores0 = np.repeat([scores0], scores.shape[0], axis = 0)
    APCS = scores - scores0
    
    return APCS

def APCS_pipeline(source_df, receptor_df, receptor_df_pca, pfas_vars, source_name, units_col, num_pcs):
    '''
    Pipeline to create APCS for a given source on the receptor well df
    '''
    
    # Get pfas_vars available
    source_pfas_vars = get_pfas_vars(df = source_df,
                                pfas18 = pfas_vars)
    
    # Filter all dataframes down to only available pfas among source and receptor
    pfas_in_common = list(set(source_pfas_vars).intersection(pfas_vars))
    filt_receptor_df_pca = receptor_df_pca[pfas_in_common]
    

    source_df_pca = normalize_over_partitions(df = source_df, 
                                              pfas_vars = source_pfas_vars,
                                              units_col = units_col,
                                              scaler_fn = 'Zscore')

    # recreate source PCA
    source_pca = PCA(n_components = num_pcs[source_name])
    pca1 = source_pca.fit_transform(source_df_pca[pfas_in_common])

    # Transform receptor df
    source_transformation = source_pca.transform(filt_receptor_df_pca.copy())

    # Transform into Absolute Principal Scores
    source_APCS = create_APCS(source_pca, receptor_df[pfas_in_common])

    source_APCS_df = pd.DataFrame(source_APCS)
    for col in source_APCS_df.columns:
        source_APCS_df.rename(columns = {col : f'{source_name}_{col}'}, inplace = True)
    
    
    return source_APCS_df

def get_redundant_pairs(df):
    '''Get diagonal and lower triangular pairs of correlation matrix'''
    pairs_to_drop = set()
    cols = df.columns
    for i in range(0, df.shape[1]):
        for j in range(0, i+1):
            pairs_to_drop.add((cols[i], cols[j]))
    return pairs_to_drop

def get_top_abs_correlations(df, corr_cutoff):
    au_corr = df.corr().abs().unstack()
    labels_to_drop = get_redundant_pairs(df)
    au_corr = au_corr.drop(labels=labels_to_drop).sort_values(ascending=False)
    corr_output = pd.DataFrame(au_corr).reset_index()
    corr_output = corr_output[corr_output[0] > corr_cutoff]
    return corr_output

def fit_model(concat_df, predictor_list, pfas_vars, n_iterations):
    model2_info = {}
    for iteration in range(n_iterations):
        model2_info[iteration] = {}

        for pfas in pfas_vars:
            model2_info[iteration][pfas] = {}

            rs = resample(concat_df, n_samples = concat_df.shape[0], random_state = iteration).copy()
            X = rs[predictor_list]
            y = rs[pfas].copy()

            lm = PoissonRegressor().fit(X, y)
            model2_info[iteration][pfas]['intercept'] = lm.intercept_
            model2_info[iteration][pfas]['coef'] = lm.coef_
            model2_info[iteration][pfas]['r2'] = lm.score(X, y)
            model2_info[iteration][pfas]['MSE'] =mean_squared_error(y, lm.predict(X))

            # Check p-values. Only use p-values with relevant information
            X2 = sm.add_constant(X)
            est = sm.OLS(y, X2)
            est2 = est.fit()

            b = pd.DataFrame(est2.pvalues)

            list_relevant_components = b[b[0] < 0.1].index
            list_relevant_components = [x for x in list_relevant_components if not 'const' in x]
            model2_info[iteration][pfas]['relevant_cols'] = list_relevant_components
            
    return model2_info

def get_coefficients(concat_df, model2_info, pfas_vars,sources, n_iterations):
    pcs = {} 
    for iteration in range(n_iterations): 
        pcs[iteration] = {}

        for source in sources:

            for pfas in pfas_vars:
                relevant_cols = model2_info[iteration][pfas]['relevant_cols']

                source_cols = [x for x in relevant_cols if source in x]

                for col in source_cols:
                    idx = concat_df.columns.get_loc(col)
                    coef = abs(1 - np.exp(model2_info[iteration][pfas]['coef'][idx]))

                    pcs[iteration][col] = coef
                
    return pcs

def calc_contribution_confidence_intervals(concat_df, model2_info, pfas_vars, sources, n_iterations, removed_vars, num_pcs_dict, other_PC_profiles_names):
    # Get dictionary of coefficients
    pcs = get_coefficients(concat_df, model2_info,pfas_vars,sources, n_iterations)

    # remove all zero iterations
    iteration_contribution_dict = {}
    for iteration, value_dict in pcs.items():
        if float(sum(list(value_dict.values()))) > 0:
            iteration_contribution_dict[iteration] = value_dict

    # Solve issue of dividng effect among highly correlated variables
    full_contribution_dict = {}
    for iteration, contribution_dict in iteration_contribution_dict.items():

        for var, remove_list in removed_vars.items():
            contribution_dict[var] = contribution_dict[var] / len(remove_list)

            for remove_var in remove_list:
                contribution_dict[remove_var] = contribution_dict[var]

        sum_effect = float(sum(list(contribution_dict.values())))
        for var in contribution_dict:
            contribution_dict[var] = contribution_dict[var] / sum_effect

        full_contribution_dict[iteration] = contribution_dict

    # Create an attribution dictionary
    attribution_dict = {}

    for iteration in full_contribution_dict:
        attribution_dict[iteration] = []
        source_names = []
        for source in num_pcs_dict:

            if source.lower() != 'other':
                attribution_dict[iteration].append(round(sum([value for key, value in full_contribution_dict[iteration].items() if source.lower() in key.lower()]) * 100, 3))
                source_names.append(source)

            else:
                for pc_num, name in other_PC_profiles_names.items():
                    attribution_dict[iteration].append(round(sum([value for key, value in full_contribution_dict[iteration].items() if f'other_{str(int(pc_num[-1]) - 1)}'.lower() in key.lower()]) * 100, 3))
                    source_names.append(name)

    # Extract confidence intervals
    means = []
    ci = []
    yerror = []
    for source in enumerate(source_names):
        a = pd.DataFrame(attribution_dict).loc[source[0],:]
        intervals = st.t.interval(0.95, len(a)-1, loc=np.mean(a), scale=st.sem(a))
        lower_interval, upper_interval = round(intervals[0],2), round(intervals[1],2)

        print(f'{source[1]} attribution 95% CI:','[' , lower_interval , ',', upper_interval, '] %')

        means.append(np.average(a))
        yerror.append(np.average(a) - lower_interval)
        ci.append((lower_interval, upper_interval))
        
    return source_names, means, yerror

def calculate_average_model_score(model2_info, n_iterations, sources, pfas_vars):
    # find the average performance of the models
    # To calculate confidence interval - get sum of each iteration
    pcs = {} 
    for iteration in range(n_iterations): 
        pcs[iteration] = {}

        for source in sources:

            for pfas in pfas_vars:

                coef = model2_info[iteration][pfas]['r2']

                pcs[iteration][pfas] = coef
    
    # Find the mean and standard deviation
    score_dataframe = pd.DataFrame(pcs)

    score_dataframe[score_dataframe < 0.1] = 0

    average_score = {}
    for pfas, row in score_dataframe.iterrows():
        row = [i for i in row if i != 0]
        average_score[pfas] = (round(np.mean(row) * 100, 2), round(np.std(row) * 100, 2) )
        
    return average_score