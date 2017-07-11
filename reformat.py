import numpy as np

def newModel():
    impSrc = file("/home/sean/Documents/Research/ImpedanceData.csv","r")

    impData = impSrc.readlines()[:-3]

    for i in range(len(impData)):
        if len(impData[i]) > 1:
            impData[i] = [j for j in impData[i].split(",") if (len(j) != 0)]

    finalMatrix = map(list,zip(*(impData)))[3::6]

    impSrc.close()


    imgDst = file("/home/sean/Documents/Research/ImpedancePS.csv","w")

    for i in finalMatrix:
        imgDst.write((",".join(i))+"\n")

    imgDst.close()


def uterusData():
    rSrc = file("kidney/Kidney_CC1cap.csv",'r')
    thetaSrc = file("kidney/Kidney_CC1con.csv",'r')

    r = [i.split(",") for i in rSrc.readlines()]
    theta = [i.split(",") for i in thetaSrc.readlines()]

    rSrc.close()
    thetaSrc.close()

    capacitance = []
    conductance = []

    f = np.logspace(4,8,401)
    #print f

    for i in range(len(r)):
        #tmpCap = []
        #tmpCon = []
        #for j in range(len(r[i])):
            #Z = float(r[i][j])*np.cos(np.radians(float(theta[i][j]))) + 1j*float(r[i][j])*np.sin(np.radians(float(theta[i][j])))
            #tmpCap.append(str(float(1/np.real(Z))))
            #tmpCon.append(str(1/(np.imag(Z)*2*3.1415926535*f[j])))
        #capacitance.append(tmpCap)
        #conductance.append(tmpCon)

        capDst = file("kidney/CC1cap%d.csv"%(i+1),'w')
        conDst = file("kidney/CC1con%d.csv"%(i+1),'w')

        capDst.write(','.join(r[i]))
        conDst.write(','.join(theta[i]))

        capDst.close()
        conDst.close()


def waterData(path):
    outpt = file("water_test_data/node1.csv","a")

    final = []
    for trial in range(1,11):
        current_file = file(path%(trial,"cap"),"r")
        this_row = [float(i) for i in current_file.readline().split(',')]
        current_file.close()

        current_file = file(path%(trial,"con"),"r")
        this_line = current_file.readline().split(',')
        for val in range(len(this_line)):
            this_row[val] += 1j*float(this_line[val])

        final.append(this_row)

    for line in final:
        outpt.write(','.join([str(i) for i in line])+"\n")

    outpt.close()

def cancerData(path):
    outpt = file("kidney/KidneyCC1.csv","w")

    f = np.logspace(4,8,401)

    #r
    current_file = file(path%("cap"),"r")
    Rs = [[float(i) for i in j.split(',')] for j in current_file.readlines()]
    current_file.close()

    #theta
    current_file = file(path%("con"),"r")
    lines = current_file.readlines()
    for line in range(len(lines)):
        THETAs = lines[line].split(',')
        for val in range(len(THETAs)):
            Z = Rs[line][val]*np.cos(np.radians(float(THETAs[val]))) + 1j*Rs[line][val]*np.sin(np.radians(float(THETAs[val])))
            Y = 1/Z

            capacitance = np.imag(Y)/(2*np.pi*f[val])
            conductance = np.real(Y)

            Rs[line][val] = capacitance + 1j*conductance

    for line in Rs:
        outpt.write(','.join([str(i) for i in line])+"\n")

    outpt.close()

def UterusCOXNew(path):
    outpt = file("uterus_other/BB1.csv","w")

    current_file = file(path%("cap"),"r")
    Rs = [[float(i) for i in j.split(',')] for j in current_file.readlines()]
    current_file.close()

    #theta
    current_file = file(path%("con"),"r")
    lines = current_file.readlines()
    for line in range(len(lines)):
        THETAs = lines[line].split(',')
        for val in range(len(THETAs)):
            Rs[line][val] += 1j*float(THETAs[val])

    for line in Rs:
        outpt.write(','.join([str(i) for i in line])+"\n")

    outpt.close()

def soundData():
    inName = "8JulySound/piano/E2-0%d.txt"
    outName = "8JulySound/piano/E2.csv"

    outpt = file(outName,"w")

    for i in range(10):
        tmp = []
        inp = file(inName%i,"r")
        for line in inp.readlines():
            if line[0] == ";":
                continue
            tmp.append(line.split()[1])
        inp.close()

        outpt.write(','.join(tmp)+'\n')
    outpt.close()

#waterData("water_test_data/NaCl-50/NaCl_50%d%s.csv")
#cancerData("kidney/Kidney_CC1%s.csv")
#UterusCOXNew("uterus_other/Uterus_BB1%s.csv")
soundData()
