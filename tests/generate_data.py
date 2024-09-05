from data_preprocessing.support_functions import (generate_county_data_csv,generate_deaths_by_age_group,
                                                  generate_hsa_mapped_county_icu_data, generate_hsa_mapped_county_hosp_data,
                                                  generate_county_infection_estimates_csv, generate_county_info_csv,
                                                  generate_county_infections_csv)

generate_county_data_csv('cases')
generate_county_data_csv('hospitalizations')
generate_county_data_csv('deaths')
generate_county_data_csv('icu')




generate_hsa_mapped_county_hosp_data()
generate_hsa_mapped_county_icu_data()
'''

#generate_deaths_by_age_group()
#generate_hosps_by_age_group()
#generate_cases_by_age_group()

'''