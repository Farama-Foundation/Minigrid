import pickle5 as pickle

with open("interspersed.pickle", "rb") as f:
    values = pickle.load(f)

for directionSplit in range(5, 11):
    dirSplit = round(directionSplit/10, ndigits=1)
    print("DirSplit: " + str(dirSplit))
    for density in range(1, 20):

        dens = round(density * 0.05, ndigits=2)

        avgSpeed = round(values[directionSplit - 5][density - 1][0], ndigits=5)
        avgVolume = round(values[directionSplit - 5][density - 1][1], ndigits=5)

        print("Density: " + str(dens) + " avgSpeed: " + str(avgSpeed) + " avgVolume: " + str(avgVolume))
