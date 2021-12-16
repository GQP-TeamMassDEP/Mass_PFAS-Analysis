# threshold at which compounds are dropped
non_detect_threshold = 0.90

# Path of the df from extraction (Assumed to be .csv)
extracted_df_path = 'c:\\Users\\dcher\\OneDrive\\Desktop\\repos\\PFAS-Analysis\\/data/Extracted lab report data/Extracted PFAS data - 2021-11-17 02_47_56.711436'

# Folder structure that will output ROS results
ros_folder = 'c:\\Users\\dcher\\OneDrive\\Desktop\\repos\\PFAS-Analysis\\scripts\\imputation/extracted_data'

# Disposal sites information (Assumed to be .csv)
disposal_sites_dict = { 'file_location' : '../../data/disposal_sites/PFAS_Sites_2021-11-07',
                        'address_col' : 'Address',
                        'town_col' : 'Town',
                        'state_col' : 'State'}

# Name of output file
disposal_sites_output = '../../data/disposal_sites/PFAS_Sites_2021-11-07_geocoded'

# PFAS compounds list
pfas_dict = { 'file_location' : '../../data/Extracted lab report data/PFAS_compounds.csv', # should be csv
            'pfas_filter_col' : 'PFAS6',
            'acronym_col' : 'Acronym'}


