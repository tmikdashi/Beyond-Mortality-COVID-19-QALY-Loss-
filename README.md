# covid19-qaly-loss

This repository provides the data, code, and scripts used to prepare the analysis and generate the figures reported in the main manuscript and supplementary materials.

**Overview** 

#### _/Data_
- **county_time_data_all_dates**: Cleaned and compiled COVID-19 case data by Bilinski et al. 
  - Source: Bilinski AM, Salomon JA, Hatfield LA. Adaptive metrics for an evolving pandemic: A dynamic approach to area-level COVID-19 risk designations. Proc Natl Acad Sci U S A 120, e2302528120 (2023). 
  - Data is available at: https://github.com/abilinski/AdaptiveRiskMetrics  
- **County_names_HSA_number.csv**: Mapping of counties to Health Service Areas (HSAs), FIPS codes, and HSA numbers.
  - Adapted from https://seer.cancer.gov/seerstat/variables/countyattribs/hsa.html 
- **Covidestim_state_infections.csv**: Weekly state-level median estimates of incident infections generated using the covidestim package.
  - Source: Russi M, Chitwood M. covidestim: Real-time Bayesian forcasting of R_t. (2022). 
  - Data is available at: https://covidestim.org/ and https://github.com/covidestim 
- **HHS_COVID_Reported_Hospital_ICU_Capacity.csv**: COVID-19 hospital and ICU admission data from the U.S. Department of Health and Human Services.
  - Source: https://public-data-hub-dhhs.hub.arcgis.com/pages/Hospital%20Utilization 
- **Provisional_COVID-19_Deaths_by_Sex_and_Age.csv** CDC- NCHS data on provisional COVID-19 deaths, by sex, age group, and jurisdiction
  - Data is available at: https://data.cdc.gov/NCHS/Provisional-COVID-19-Deaths-by-Sex-and-Age/9bhg-hcku/about_data 
- **CDC_life_table_female_2019_USA.csv**: CDC life table for females in the United States, 2019.
  - Reference:  Arias E, Xu JQ. United States life tables, 2019. National Vital Statistics Reports; vol 70 no 19. Hyattsville, MD: National Center for Health Statistics. 2022. DOI: https://dx.doi.org/10.15620/cdc:113096.  
  - Data is available at: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/70-19/Table03.xlsx 
- **CDC_life_table_male_2019_USA.csv**: CDC life table for males in the United States, 2019
  - Reference:  Arias E, Xu JQ. United States life tables, 2019. National Vital Statistics Reports; vol 70 no 19. Hyattsville, MD: National Center for Health Statistics. 2022. DOI: https://dx.doi.org/10.15620/cdc:113096.
  - Data is available at: https://ftp.cdc.gov/pub/Health_Statistics/NCHS/Publications/NVSR/70-19/Table02.xlsx 

#### _/Tests_

To replicate the figures in the manuscript and supplementary materials, run the two scripts located in the `/tests` folder:
- `generate_data.py`: Prepares all necessary data for analysis.
- `test_model.py`: Runs the main model and generates all figures for the manuscript and supplementary materials.

**Step 1: Data Preparation** (`/tests/generate_data`) 
This script prepares all data required for analysis. Key processing steps include:

- Estimating weekly county-level infections using reported county-level cases and state-level estimated infections from covidestim 
- Preparing weekly county-level hospital admissions, ICU occupancy and death data 
- Preparing weekly county-level vaccination sensitivity analysis dataset for long COVID based on infections data 
- Preparing weekly HSA-mapped hospitalization admission and icu occupancy data 
- Preparing age-stratified data for hospital admissions and deaths
	
The functions used to prepare this data and their description is provided in `/data_prepocessing/support_functions`. Outputs will be saved in `/csv_files` directory 

**Step 2: Data Analysis and Figure Generation** (`/tests/test_model`)
This script performs the full analysis and reproduces all figures presented in the manuscript and supplementary materials. Figures are saved to the /figs directory 

#### /Parameters 

This folder contains model parameters used in the analysis. These correspond to Tables 1 and 2 in the manuscript, which provide detailed descriptions and sources for each parameter. 

#### /Data_preprocessing/support_functions 

Helper functions used throughout data preparation. Referenced in both main scripts and modularized for reuse.

All data analysis was performed in Python (PyCharm 2023.3.4 (Community Edition). If Python is already installed, there should be no additional install time on on the computer.