Notes regarding imputation pipeline:

General inputs to the process are in constants.py. This should be updated to your specific machine. Other inputs to the process occur along the line in the jupyter notebooks. Descriptions regarding the inputs are commented near the inputs within the notebook.

The code works on the assumption that the lab reports have been extracted using the Lab Report Extraction Tool. All functions referenced in the code are housed in imputation_utils.py. 

1. impute_extracted.ipynb: The code focuses on taking the extracted data and imputing the non-detects that pop up within the sample. 

2. create_input_files.ipynb: The code focuses on creating the necessary files and file modifications in order to create the modeling datasets for the different models. 

3. create_well_exposure_model_data.ipynb: The code focuses on creating the datasets for the well exposure models.

4a. create_source_attribution_model_data.ipynb: The code focuses on creating the datasets for the source attribution models. Since the residential profile was generated using a separate data source - that process is separate.

4b. create_source_attribution_model_data_residential.ipynb: The code focuses on creating the datasets for the residential profile in the source attribution models. 