import pickle5 as pickle

with open("DML_p=1.0.pickle", "rb") as f:
    values = pickle.load(f)

for dirSplitInt in range(5, 11):
    dirSplit = round(dirSplitInt/10, ndigits=1)
    print("DirSplit: " + str(dirSplit))
    for densityInt in range(1, 20):

        dens = round(densityInt * 0.05, ndigits=2)

        avgSpeed = round(values[dirSplitInt - 5][densityInt - 1][0], ndigits=5)
        avgVolume = round(values[dirSplitInt - 5][densityInt - 1][1], ndigits=5)

        print("Density: " + str(dens) + " avgSpeed: " + str(avgSpeed) + " avgVolume: " + str(avgVolume))
