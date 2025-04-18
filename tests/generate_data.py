from data_preprocessing.support_functions import (generate_county_data_csv,generate_deaths_by_age_group,
                                                  generate_hsa_mapped_county_icu_data, generate_hsa_mapped_county_hosp_data,
                                                   generate_county_info_csv,distribute_infections_in_counties,
                                                   generate_deaths_by_age_group,generate_state_divided_county_infections_csv,
                                                  generate_state_cases_infections_factor,generate_infections_from_cases,
                                                  generate_hosps_by_age_group,generate_symptomatic_infections_vax)

# Preparing weekly county-level infections estimates
generate_county_data_csv('cases')
generate_county_info_csv()
distribute_infections_in_counties()
generate_state_divided_county_infections_csv()
generate_state_cases_infections_factor()
generate_infections_from_cases()

# Preparing weekly county-level hospital admissions, ICU occupancy and death data
generate_county_data_csv('hospitalizations')
generate_county_data_csv('deaths')
generate_county_data_csv('icu')

# Preparing weekly county-level vaccinated infections data
generate_symptomatic_infections_vax()

# Preparing weekly hsa-mapped hospitalization admission and icu occupancy data
generate_hsa_mapped_county_hosp_data()
generate_hsa_mapped_county_icu_data()

# Preparing hospital admission and deaths data by age group
generate_hosps_by_age_group()
generate_deaths_by_age_group()

