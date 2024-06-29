from src._constants import GraModel, graSensitivity, MagModel, magSensitivity

def gendata():

    model = GraModel(0, 1000, 35 * 1000, 
                        0, 1000, 20 * 1000, 
                        0, 1000, 35 * 1000,
                        0, 1000, 20 * 1000,
                        0, 1000, 15 * 1000)
    model.property[8:13, 7:12, 3:8] = 1
    model.property[22:27, 7:12, 5:10] = -1.5
    model.forward()
    data = []
    data.append(model.anomaly.T)

    model = MagModel(0, 500, 30 * 500, 
                        0, 500, 20 * 500,  # start, step, and end location of measure system(m)
                        0, 500, 30 * 500,
                        0, 500, 20 * 500,
                        0, 500, 15 * 500)  # start, step, and end location of underground cubes(m)
    # set the properties of underground space, here we set a cuboid model(A/m in mag, g/cm^3 in gra)
    model.property[7:13, 7:13, 3:9] = 1
    model.property[17:23, 7:13, 6:12] = -1
    # compute anomaly(nT in mag, mGal in gra) 
    model.forward()
    data.append(model.anomaly.T)

    return data