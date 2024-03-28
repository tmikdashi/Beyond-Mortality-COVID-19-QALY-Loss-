from data_preprocessing.support_functions import (generate_county_data_csv, generate_hosps_by_age_group,
                                                  generate_hsa_mapped_county_hosp_data, generate_deaths_by_age_group,
                                                  generate_correlation_matrix_total, generate_correlation_matrix_total_per_capita)

#generate_county_data_csv('cases')
#generate_county_data_csv('hospitalizations')
#generate_county_data_csv('deaths')

#generate_hsa_mapped_county_hosp_data()
generate_deaths_by_age_group()
generate_hosps_by_age_group()


generate_correlation_matrix_total()
generate_correlation_matrix_total_per_capita()
