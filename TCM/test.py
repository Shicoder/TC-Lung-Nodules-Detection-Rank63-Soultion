
__author__ = 'shi'
import pandas as pd

# patient_space=pd.read_csv('../DSB2017/data/ndsb3_extracted_images/patient_spacing.csv')
# print( patient_space.shape)
# # patient_space.set_index(['patient_id'],inplace=True)
# # patient_space = patient_space.drop_duplicates()
# print(patient_space.shape)
# print(patient_space.head())
# print(patient_space.loc[patient_space.patient_id=='LKDS-00031'])
# print(patient_space.sort_values(by='spacing_x',ascending = False))
a = [1,2,3]
aa = pd.DataFrame(a)
print(aa)
bb = aa.copy()
print(bb)
bb[0]=2
print(bb)
print(aa)