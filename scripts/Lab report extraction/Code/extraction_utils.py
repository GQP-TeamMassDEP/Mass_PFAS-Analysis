# -*- coding: utf-8 -*-

##
##!pip install ghostscript
##!pip install camelot-py[cv]
##!apt install ghostscript python3-tk
##
##
##from ctypes.util import find_library
##print("Ghostscript library:", find_library("gs"))


import ghostscript
import camelot
import pandas as pd
import glob
import os
import re
import warnings

#!pip install PyPDF2
import PyPDF2 

import datetime



"""# Lab report identification
---
#### Identify all lab reports in folders.
#### Create table of report file info.
#### Determine and record testing lab for each report.
---


"""


def determine_report_type(report_file, *args):

    # creating a pdf file object 
    pdfFileObj = open(report_file, 'rb') 
        
    # creating a pdf reader object 
    try:
      pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False) 
    except:
      pdfFileObj.close() 
      return {'error':'file read error'}
    
    try:
      num_pages = pdfReader.numPages
    except:
      pdfFileObj.close() 
      return {'error':'no pages read'}

    all_pages = []
    for i in range(num_pages):
      # creating a page object 
      try:
        pageObj = pdfReader.getPage(i) 
      except:
        return {'error':'page read error'}

      # extracting text from page 
      pageText = pageObj.extractText()

      if 'debug' in args:
        print("page:",i)
        print(pageText) 
      
      all_pages.append(pageText)

    pdfFileObj.close() 

    if all([x=="" for x in all_pages]):
      return {'error': 'Non-text PDF'}

    report_type = ""
    if any('www.sgs.com' in page for page in all_pages):
      if any('Table of Contents' in page for page in all_pages):
        report_type = 'sgs'
      elif any('FINAL LAB REPORT' in page for page in all_pages):
        report_type = 'sgs-final'
    if any('SGS LabLink' in page for page in all_pages):
        report_type = 'sgs-no-TOC'
    if any('CON-TEST' in page for page in all_pages):
        report_type = 'con-test'  
    if any('con-test' in page for page in all_pages):
        report_type = 'con-test'
    if any('www.alphalab.com' in page for page in all_pages):
        report_type = 'alpha'

    if (report_type != ""):
      return {'lab':report_type}
    else:
      return {'error': 'could not determine lab'}



def strip_folder_info(source):
    
    RTN_regex = '\\d-\\d{7}'

    m = re.search(RTN_regex, source)
    RTN = m.group()

    other_txt = source.strip(RTN)

    other_list = other_txt.split('-')
    other_list = [x.replace('/','').strip() for x in other_list if x!=' ']

    if len(other_list) == 2:
      town = other_list[0].title()
      disposal_site_name = other_list[1].replace('PFAS','').strip() 

    return {'RTN':RTN,
            'town':town,
            'disposal_site_name':disposal_site_name}



"""# Lab Report Sample Results Page(s) Location
---
#### For all identified reports (without read errors):
#### Execute lab-specific functions to location results table in PFD and record page numbers in existing report file dataframe.
---


"""



def verify_filenames(df):
    print("Total filenames:", df.shape[0])
    invalid_filenames = []

    for i,x in df.iterrows():
      filename = x.path + x.folder + x.report

      if os.path.exists(filename):
        pass
      else:
        invalid_filenames.append(filename)

    if invalid_filenames:
      print("Invalid filenames:")
      print(invalid_filenames)
    else:
      print('All filenames are valid')



def SGS_find_results_pages(report_file, *args):
    # SGS 
    # Extract Table of Contents to identify section pages


    pdfFileObj = open(report_file, 'rb') 
    try:
      pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False) 
    except:
      pdfFileObj.close() 
      return {'error':'file read error'}

    try:
      num_pages = pdfReader.numPages
    except:    
      pdfFileObj.close() 
      return {'error':'num pages read error'}
    
    if 'debug' in args:
      print("num_pages:", num_pages) 

    # find table of contents
    toc_page = 0
    for i in range(0, num_pages):
      pageObj = pdfReader.getPage(i) 
      pageText = pageObj.extractText()

      if ("Table of Contents" in pageText):
        toc_page = i + 1
        if 'debug' in args:
          print('Table of Contents: page',toc_page)
        break
    
    pdfFileObj.close()

    if toc_page == 0:  
      return {'error':'no table of contents found'}
    
    # extract table of contents
    tables = camelot.read_pdf(report_file, pages=str(toc_page),
                              flavor='stream')

    parsing_accuracy = tables[0].parsing_report['accuracy']
    if 'debug' in args:
      print("parsing accuracy:", parsing_accuracy)

    if (parsing_accuracy < .7):
      return {'error':'low parsing accuracy'}

    df = tables[0].df

    if 'debug' in args:
      display(df)

    df.loc[len(df)] = df.columns
    df.columns = ["Section", "Page", "dummy"]

    df.drop("dummy", axis=1, inplace=True)
    df = df.loc[df['Page'] != '',]

    # Find sample results section in table of contents (adjust for page numbering)
    start_page = int(df.loc[df['Section'].str.contains("Sample Results", na=False),'Page']) + 1
    end_page = int(df.loc[df['Section'].str.contains("Misc. Forms", na=False),'Page']) - 1
  
    if 'debug' in args:
      print("start of section:", start_page, type(start_page))
      print("end of section:", end_page, type(end_page))

    if start_page and end_page:
      results_pages = list(range(start_page, end_page+1))    # add 1 due to range exclusion
      # convert list to a comma separated string
      results_pages = [str(int(x)) for x in results_pages]
      pages_str = ",".join(results_pages)
      return {'results_pages':pages_str}
    else:
      return {'error':'no results pages found'}



def CON_TEST_find_results_pages(report_file, *args):
  # Con-test Analytical Laboratory
  # Extract Table of Contents to identify section pages
      
    # extract table of contents
    tables = camelot.read_pdf(report_file, pages='2',
                              flavor='stream')

    table = tables[0]
    df = table.df

    if 'debug' in args:
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}

    df.loc[len(df)] = df.columns
    df.columns = ["Section", "Page"]

    # Find sample results section in table of contents
    start_page = int(df.loc[df["Section"]=="Sample Results", "Page"])
    end_page = int(df.loc[df["Section"]=="Sample Preparation Information", "Page"]) - 1

    if 'debug' in args:
      print("start of section:", start_page)
      print("end of section:", end_page)

    if start_page and end_page:
      results_pages = list(range(start_page, end_page+1))    # add 1 due to range exclusion
      # convert list to a comma separated string
      results_pages = [str(int(x)) for x in results_pages]
      pages_str = ",".join(results_pages)
      return {'results_pages':pages_str}
    else:
      return {'error':'no results pages found'}



def ALPHA_find_results_pages(report_file, *args):
  # Alpha Analytical 
  # Identify pages containing sample results

    # creating a pdf file object 
    pdfFileObj = open(report_file, 'rb') 
        
    # creating a pdf reader object 
    try:
      pdfReader = PyPDF2.PdfFileReader(pdfFileObj, strict=False) 
    except:
      pdfFileObj.close() 
      return {'error':'file read error'}
        
    try:
      num_pages = pdfReader.numPages
    except:
      pdfFileObj.close() 
      return {'error':'num pages read error'}

  # # adjust for possible cover page(s) before report pages
  # start_page = 0
  # for i in range(num_pages):
  #   if ('Page 1 of' in pdfReader.getPage(i).extractText()):
  #     if start_page == 0:
  #       start_page = i

    results_pages = []
    for i in range(num_pages):
      # creating a page object 
      pageObj = pdfReader.getPage(i) 
          
      # extracting text from page 
      pageText = pageObj.extractText()

      if 'debug' in args:
        print('\n\n----------------')
        print('page (0 indexed):', i)
        print(pageText[0:50])

      if ("SAMPLE RESULTS" in pageText) and ("Perfluorinated Alkyl Acids" in pageText) and ("Extracted Internal Standard" not in pageText):
        results_pages.append(i)
        if 'debug' in args:
          print('Identified results - page:',i)
        
    # closing the pdf file object 
    pdfFileObj.close() 
  
    if results_pages:
      # adjust for page numbering
      results_pages = [x+1 for x in results_pages]
      # convert list to a comma separated string
      results_pages = [str(int(x)) for x in results_pages]
      pages_str = ",".join(results_pages)
      return {'results_pages':pages_str}
    else:
      return {'error':'no results pages found'}



"""# Lab Report Sample Results Data Extraction
---
#### For all identified reports with results pages identified:

#### Execute lab-specific functions for data table extraction on all identified pages.
#### Process data columns for consistent naming and formats.
#### Obtain standardized PFAS compound names, acronyms and CAS numbers for each table.
#### Append all to master dataframe and export.  

---


"""



def get_PFAS_list(filename):

    tables = camelot.read_pdf(filename, pages='2', flavor='stream', 
                              edge_tol=0,
                              row_tol=10
                              )

    df = tables[0].df

    idx_start = df.index[df[0] == 'Parameter'].tolist()[0]
    idx_end = df.index[df[0] == '* also available by Method 537.1 for drinking water samples'].tolist()[0]

    df = df.rename(columns=df.loc[idx_start])
    df = df.rename(columns={'Parameter':'Compound'})

    df = df.loc[(idx_start+1):(idx_end-1),]

    df = df.loc[df['Compound']!='', ]
    df = df.loc[df['Acronym']!='', ]

    df = df[['Compound','Acronym', 'CAS']]
    df = df.reset_index(drop=True)

    df['Compound'] = df['Compound'].str.replace('\n', ' ')
    df['Acronym'] = df['Acronym'].str.replace('*', '')
    df['Acronym'] = df['Acronym'].str.replace('\n', ' ')
    df['Acronym'] = df['Acronym'].str.strip()

    return df



def SGS_extract_results(report_file, page, PFAS_names, *args):
    # SGS Analytical Laboratory
    # Extract report table, Sample Results - Analytes 
    # Semivolatile Organic Compounds by - LC/MS-MS

    page = str(int(page))

    # verify page contains PFAS sample results
    tables = camelot.read_pdf(report_file, pages=page,
                              # table_areas = ['40,520,565,160'],
                              flavor='stream')
    table = tables[0]
    df = table.df

    if 'debug' in args:
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}

    if not ('PERFLUOROALKYLCARBOXYLIC ACIDS' or 
            'PERFLUOROALKYLSULFONATES ACIDS' or 
            'PERFLUOROALKYLSULFONIC ACIDS' or
            'PERFLUOROOCTANESULFONAMIDOACETIC ACIDS' or 
            'NEXT GENERATION PFAS ANALYTES') in df.to_string():
      return {'error':'not sample results data'}


    # extract table with settings for upper summary info
    tables = camelot.read_pdf(report_file, pages=page,
                              # row_tol=0,
                              # edge_tol=50,
                              table_areas = ['20,760,562,590'],  # top section
                              flavor='stream')
    
    table = tables[0]
    df = table.df

    if 'debug' in args:
      print('1st table extraction')
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}


    matrix = df.loc[df[0]=='Matrix:',1].values
    if matrix.size > 0:
      matrix = matrix[0]
    else:
      matrix = ''
    
    try:
      sample_id_1 = df.loc[df[0]=='Client Sample ID:',1].values[0]
      sample_id_2 = df.loc[df[0]=='Client Sample ID:',2].values[0]
      sample_id = sample_id_1 + sample_id_2
    except:
      sample_id = ''

    try:
      date_sampled = df.loc[df[2]=='Date Sampled:',3].values[0]
    except:
      date_sampled = ''

    if 'debug' in args:
      print('sample_id:', sample_id)
      print('date_sampled:', date_sampled)


    # extract table with settings for upper summary info (2nd half)
    tables = camelot.read_pdf(report_file, pages=page,
                              # row_tol=0,
                              # edge_tol=50,
                              table_areas = ['20,600,562,500'],  
                              flavor='stream')                        

    table = tables[0]
    df = table.df

    if 'debug' in args:
      print()
      print('2nd table extraction')
      print("parsing accuracy:", table.accuracy)
      display(df)


    dilution_idx = df.index[df[2].str.contains('DF')].tolist()
    if dilution_idx:
      dilution_idx = dilution_idx[0] + 1
      dilution = df.loc[dilution_idx, 2]
    else:
      dilution = ''



    # extract with settings for results table columns
    tables = camelot.read_pdf(report_file, pages=page,  flavor='stream',
                              flag_size=True,
                              table_areas = ['40,520,565,160'],
                              columns=['247, 291, 330, 360, 390, 420'])
                              # columns=['247, 291, 330, 346, 380, 404, 459'])

    table = tables[0]
    df = table.df

    if 'debug' in args:
      print()
      print('2nd table extraction')
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}

    if df.loc[df[0].str.contains('Compound'),].shape[0] == 0:
      return {'error':'not sample results data'}

    # check whether this table contains an MCL and/or MDL column
    has_MCL = 'MCL' in df.to_string()
    has_MDL = 'MDL' in df.to_string()

    # select specific rows containing sample results
    idx_start = df.index[df[0].str.contains('Compound')].tolist()
    if idx_start:
      idx_start = idx_start[0]
      df = df.loc[(idx_start+1):,]

    idx_end = df.index[df[0].str.contains('Recoveries')].tolist()
    if idx_end:
      idx_end = idx_end[0]
      df = df.loc[:(idx_end-1),]

    df = df.loc[df[1]!="",]  
    df.reset_index(drop=True, inplace=True)

    # setup function to identify CAS from compound column string
    def get_CAS(row):
        source = row
        CAS_regex = '[1-9]{1}[0-9]{1,5}-\\d{2}-\\d'
        m = re.search(CAS_regex, source)
        match = m.group()
        other = source.strip(match)
        other = other.replace("\n","")
        return pd.Series({'CAS':match, 'Compound_orig':other})

    df1 = df[0].apply(get_CAS)
    df = pd.concat((df, df1), axis=1)
    df = df.rename(columns={'CAS':'CAS_orig'})

    def get_PFAS_info_by_CAS(CAS):
        if CAS in PFAS_names['CAS'].to_string():
          matched = PFAS_names.loc[PFAS_names['CAS']==CAS,]
          if matched.shape[0] >= 1:
            return matched.iloc[0]
        else:
          return pd.Series({'Compound':'',
                            'Acronym':'',
                            'CAS':''})
  
    df1 = df['CAS_orig'].apply(get_PFAS_info_by_CAS)
    df = pd.concat((df, df1), axis=1)
    df = df.loc[df['CAS']!='',]
    df.reset_index(drop=True, inplace=True)

    if 'debug' in args:
      display(df)

    df['lab'] = 'SGS'
    df['date_sampled'] = date_sampled
    df['sample_id'] = sample_id
    df['Matrix'] = matrix
    df['DF'] = dilution
    

    if not 'Units' in df.columns:
      df['Units'] = ''

    if not 'RL' in df.columns:
      df['RL'] = ''

    if has_MCL and not has_MDL:
      df['Result'] = df[1]
      df['MCL'] = df[2]
      df['RL'] = df[3]
      df['MDL'] = ''
      df['Units'] = df[4]
    elif has_MDL and not has_MCL:
      df['Result'] = df[1]
      df['MCL'] = ''
      df['RL'] = df[2]
      df['MDL'] = df[3]
      df['Units'] = df[4]
    elif has_MDL and has_MCL:
      df['Result'] = df[1]
      df['MCL'] = df[2]
      df['RL'] = df[3]
      df['MDL'] = df[4]
      df['Units'] = df[6]
    else:
      df['Result'] = df[1]
      df['MCL'] = ''
      df['MDL'] = ''
      df['RL'] = df[2]
      df['Units'] = df[3]

    if 'debug' in args:
      print()
      print('has_MCL:', has_MCL)
      print('has_MDL:', has_MDL)

    # strip out superscript markers
    df['Result'] = df['Result'].str.replace(r'<s>[a-z]<\/s>','')

    df = df[['lab','date_sampled','sample_id','Matrix','DF','CAS','Compound','Acronym','Result','RL','MCL','MDL','Units']]

    df = df.loc[-pd.isna(df['Acronym']),]
    df = df.reset_index(drop=True)

    return {'results':df}



def CON_TEST_extract_results(report_file, page, PFAS_names, *args):
    # Con-test Analytical Laboratory
    # Extract report table, Sample Results - Analytes 
    # Semivolatile Organic Compounds by - LC/MS-MS

    page = str(int(page))

    # extract table with settings for upper summary info
    tables = camelot.read_pdf(report_file, pages=page,
                              table_areas = ['20,720,570,660'],
                              flavor='stream'
                              )
    table = tables[0]
    df = table.df

    if 'debug' in args:
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}

    sample_id = df.loc[df[0].str.contains('Sample ID:'),0].values
    if sample_id.size > 0:
      sample_id = sample_id[0].replace('Sample ID:','').strip()
    else:
      sample_id = ''

    date_sampled = df.loc[df[1].str.contains('Sampled:'),1].values
    if date_sampled.size > 0:
      date_sampled = date_sampled[0].replace('Sampled:','')
      date_sampled = re.sub('\\d{1,2}\\:\\d{1,2}', repl='', string=date_sampled).strip()
    else:
      date_sampled = ''

    address = df.loc[df[0].str.contains('Field Sample #:'),0].values
    if address.size > 0:
      address = address[0].replace('Field Sample #:', '').strip()
    else:
      address = ''

    town = df.loc[df[0].str.contains('Project Location:'),0].values
    if town.size > 0:
      town = town[0].replace('Project Location:', '').strip()
    else:
      town = ''

    matrix = df.loc[df[0].str.contains('Sample Matrix:'),0].values
    if matrix.size > 0:
      matrix = matrix[0].replace('Sample Matrix:', '').strip()
    else:
      matrix = ''

    address = address + ', ' + town


    # extract results table rows
    tables = camelot.read_pdf(report_file, pages=page,
                              flavor='stream',
                              row_tol=10,
                              table_areas = ['20,630,570,80']
                              )

    table = tables[0]
    df = table.df

    if 'debug' in args:
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}


    # select specific rows containing sample results
    idx_start = df.index[df[0].str.contains('Analyte')].tolist()

    if idx_start:
      idx_start = idx_start[0]
      column_names = df.loc[idx_start]
      df = df.loc[(idx_start+1):,]
    else:
      return {'error':'Results not found'}

    idx_end = df.index[df[0].str.contains('Surrogates')].tolist()
    if idx_end:
      idx_end = idx_end[0]
      df = df.loc[:(idx_end-1),]

    df = df.rename(columns=column_names)
    
    df = df.loc[df['Analyte']!="",]  
    df.reset_index(drop=True, inplace=True)


    for i,x in PFAS_names.iterrows():
      idx = df.loc[df['Analyte'].str.contains(x['Acronym'].replace(' ','')),].index
      if idx.any:
        df.loc[idx, 'Compound'] = x['Compound']
        df.loc[idx, 'Acronym'] = x['Acronym']
        df.loc[idx, 'CAS'] = x['CAS']

    df['lab'] = 'Con-Test'
    df['address'] = address
    df['date_sampled'] = date_sampled
    df['sample_id'] = sample_id
    df['Matrix'] = matrix
    
    if 'Results' in df.columns:
      df = df.rename({'Results': 'Result'}, axis=1) 
    else:
      return {'error':'Results not found'}

    if 'MCL' not in df.columns:
      df['MCL'] = ''

    if 'Dilution' in df.columns:
      df = df.rename({'Dilution': 'DF'}, axis=1) 
    else:
      df['DF'] = ''

    df = df[['lab', 'address', 'date_sampled', 'sample_id', 'Matrix',             
             'CAS','Compound','Acronym','Result','RL','MCL','Units', 'DF']]

    df = df.loc[-pd.isna(df['Acronym']),]
    df = df.reset_index(drop=True)

    return {'results':df}



def ALPHA_extract_results(report_file, page, PFAS_names, *args):
    # Alpha Analytical Laboratory
    # Extract report table, Sample Results - Analytes 
    # Semivolatile Organic Compounds by - LC/MS-MS

    page = str(int(page))

    # top section - sample info
    tables = camelot.read_pdf(report_file, pages=page,
                              table_areas = ['20,760,562,593'],
                              flavor='stream')
    table = tables[0]
    df = table.df

    if 'debug' in args:
      print("parsing accuracy:", table.accuracy)
      display(df)

    if (table.accuracy < .7):
      return {'error':'low parsing accuracy'}
    
    if df.shape[1] == 5:
      try:
        date_sampled = df.loc[df[3]=='Date Collected:',4].values[0]
      except:
        date_sampled = ''
    elif df.shape[1] == 6:
      try:
        date_sampled = df.loc[df[4]=='Date Collected:',5].values[0]
      except:
        date_sampled = ''
    else:
      date_sampled = ''

    try:  
      sample_id = df.loc[df[0]=='Client ID:',1].values[0]
    except:
      sample_id = ''

    try:  
      matrix = df.loc[df[0]=='Matrix:', 1].values[0]
    except:
      matrix = ''
    
    tables = camelot.read_pdf(report_file, pages=page,
                              row_tol = 10,
                              table_regions = ['20,650,562,82'],  # results section
                              flavor='stream')

    df = tables[0].df

    if 'debug' in args:
      print("parsing accuracy:", tables[0].parsing_report['accuracy'])
      display(df)

    idx_start = df.index[df[0] == 'Parameter'].tolist()[0]
    df = df.rename(columns=df.loc[idx_start])

    if not pd.Series(['Result', 'Parameter']).isin(df.columns).all():
      return {'error':'column formatting error'}

    df = df.loc[(idx_start+1):,]
    df = df.loc[df["Result"]!="",]
    df = df.loc[df["Parameter"]!="",]
    df = df.loc[-df["Parameter"].str.contains('Total'),]

    df.reset_index(drop=True, inplace=True)

    for i,x in PFAS_names.iterrows():
      idx = df.loc[df['Parameter'].str.contains(x['Acronym'].replace(' ','')),].index
      if idx.any:
        df.loc[idx, 'Compound'] = x['Compound']
        df.loc[idx, 'Acronym'] = x['Acronym']
        df.loc[idx, 'CAS'] = x['CAS']

    df['lab'] = 'Alpha'
    df['date_sampled'] = date_sampled
    df['sample_id'] = sample_id
    df['Matrix'] = matrix
 
    if not 'Units' in df.columns:
      df['Units'] = ''

    if not 'MDL' in df.columns:
      df['MDL'] = ''

    if not 'RL' in df.columns:
      df['RL'] = ''
      
    if 'Dilution Factor' in df.columns:
      df = df.rename({'Dilution Factor': 'DF'}, axis=1) 
    else:
      df['DF'] = ''

    df = df[['lab',
             'date_sampled',
             'sample_id',
             'Matrix',
             'CAS','Compound','Acronym','Result',
             'RL',
             'MDL',
             'Units',
             'DF']]

    df = df.loc[-pd.isna(df['Acronym']),]
    df = df.reset_index(drop=True)

    return {'results':df}

