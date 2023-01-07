import pandas as pd
import pickle5 as pickle
import numpy as np

with open("interspersed_p=1.0.pickle", "rb") as f: # DML_p=1.0.pickle, interspersed_p=0.5, interspersed_p=1.0
    values = pickle.load(f)

data = np.zeros((6 * 19, 4))
for dirSplitInt in range(5, 11):
    dirSplit = round(dirSplitInt/10, ndigits=1)
    print("DirSplit: " + str(dirSplit))
    for densityInt in range(1, 20):

        dens = round(densityInt * 0.05, ndigits=2)

        avgSpeed = values[dirSplitInt - 5][densityInt - 1][0]
        avgVolume = values[dirSplitInt - 5][densityInt - 1][1]

        data[(dirSplitInt - 5) * 19 + (densityInt - 1)][0] = dirSplit
        data[(dirSplitInt - 5) * 19 + (densityInt - 1)][1] = dens
        data[(dirSplitInt - 5) * 19 + (densityInt - 1)][2] = avgSpeed
        data[(dirSplitInt - 5) * 19 + (densityInt - 1)][3] = avgVolume
    
df = pd.DataFrame(data, columns = ['Direction Split', 'Density', 'Average Speed', 'Average Density'])
print(df.to_string())

df.to_csv('interspersed_p=1.0.csv', index=False)