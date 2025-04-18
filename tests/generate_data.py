from data_preprocessing.support_functions import (generate_county_data_csv,generate_deaths_by_age_group,
                                                  generate_hsa_mapped_county_icu_data, generate_hsa_mapped_county_hosp_data,
                                                   generate_county_info_csv,distribute_infections_in_counties,
                                                  get_dict_of_county_data_by_type, generate_deaths_by_age_group,
                                                  generate_county_infections_csv,
                                                  generate_state_cases_infections_factor,generate_infections_from_cases,
                                                  generate_hosps_by_age_group,generate_symptomatic_infections_vax)
#Generating total infections data
generate_county_data_csv('cases')
generate_county_info_csv()
distribute_infections_in_counties()
generate_county_infections_csv()
generate_state_cases_infections_factor()
generate_infections_from_cases()

#Generating total infections data
generate_county_data_csv('hospitalizations')
generate_county_data_csv('deaths')
generate_county_data_csv('icu')
generate_symptomatic_infections_vax()

generate_hsa_mapped_county_hosp_data()
generate_hsa_mapped_county_icu_data()

generate_hosps_by_age_group()
generate_deaths_by_age_group()

