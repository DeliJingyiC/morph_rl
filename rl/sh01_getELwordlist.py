from pathlib import Path
import pandas as pd
import json
import csv
import numpy as np
import matplotlib.pyplot as plt
import random
from scipy import optimize
data_path = Path("/users/PAS2062/delijingyic/project/morph/rl/dataset")
elp=pd.read_csv(
    data_path /"data.csv",
    dtype=str,
)
sublex=pd.read_csv(
    data_path /"SUBTLEXusfrequencyabove1.csv",
    dtype=str,
)

for i, row in elp['Word'].items():
    for j, rowj in sublex["Word"].items():
        if row == rowj:
            elp.loc[i,'Log_Freq_HAL'] = sublex.loc[j,'Lg10WF']

elp.to_csv(data_path / "elp_withsublex.csv",
            index=False)

elp = elp[elp["I_Mean_RT"]!=""]
elp["I_Mean_RT"] = elp["I_Mean_RT"].apply(lambda x: float(x) if type(x)==str else x)
elp["Log_Freq_HAL"] = elp["Log_Freq_HAL"].apply(lambda x: float(x) if type(x)==str else x)
y1=elp[elp["Log_Freq_HAL"].apply(lambda x: x<=2)]
y2=elp[elp["Log_Freq_HAL"].apply(lambda x:x>2 and x<=6)]
y3=elp[elp["Log_Freq_HAL"].apply(lambda x:x>6 and x<=8)]
y4=elp[elp["Log_Freq_HAL"].apply(lambda x:x>8 and x<=12)]

y5=elp[elp["Log_Freq_HAL"].apply(lambda x: x>12)]
# print(y1,len(y1))
# print(y2,len(y2))
y1_std=np.std(y1["I_Mean_RT"])
y2_std=np.std(y2['I_Mean_RT'])
y3_std=np.std(y3['I_Mean_RT'])
y4_std=np.std(y4['I_Mean_RT'])
y5_std=np.std(y5['I_Mean_RT'])
z = np.polyfit(elp["Log_Freq_HAL"], elp['I_Mean_RT'],1)
z1=np.polyfit(y1["Log_Freq_HAL"], y1['I_Mean_RT'],1)
z2=np.polyfit(y2["Log_Freq_HAL"], y2['I_Mean_RT'],1)
z3=np.polyfit(y3["Log_Freq_HAL"], y3['I_Mean_RT'],1)
z4=np.polyfit(y4["Log_Freq_HAL"], y4['I_Mean_RT'],1)
z5=np.polyfit(y5["Log_Freq_HAL"], y5['I_Mean_RT'],1)



a= np.poly1d(z,r=False,variable=["x"])
pridict_y=a(elp["Log_Freq_HAL"])


a1= np.poly1d(z1,r=False,variable=["x"])
pridict_y1=a1(y1["Log_Freq_HAL"])
predict_std1_up=pridict_y1+2*y1_std
predict_std1_low=pridict_y1-2*y1_std

a2= np.poly1d(z2,r=False,variable=["x"])
pridict_y2=a2(y2["Log_Freq_HAL"])
predict_std2_up=pridict_y2+2*y2_std
predict_std2_low=pridict_y2-2*y2_std

a3= np.poly1d(z3,r=False,variable=["x"])
pridict_y3=a3(y3["Log_Freq_HAL"])
predict_std3_up=pridict_y3+2*y3_std
predict_std3_low=pridict_y3-2*y3_std

a4= np.poly1d(z4,r=False,variable=["x"])
pridict_y4=a4(y4["Log_Freq_HAL"])
predict_std4_up=pridict_y4+2*y4_std
predict_std4_low=pridict_y4-2*y4_std

a5= np.poly1d(z5,r=False,variable=["x"])
pridict_y5=a5(y5["Log_Freq_HAL"])
predict_std5_up=pridict_y5+2*y5_std
predict_std5_low=pridict_y5-2*y5_std
plt.figure(figsize=(7.5,8))

plt.scatter(elp["Log_Freq_HAL"], elp['I_Mean_RT'],label="original")
plt.scatter(elp["Log_Freq_HAL"], pridict_y,label="predict")
plt.scatter(y1["Log_Freq_HAL"], pridict_y1,label="predict_y1")
plt.scatter(y2["Log_Freq_HAL"], pridict_y2,label="predict_y2")
plt.scatter(y3["Log_Freq_HAL"], pridict_y3,label="predict_y3")
plt.scatter(y4["Log_Freq_HAL"], pridict_y4,label="predict_y4")
plt.scatter(y5["Log_Freq_HAL"], pridict_y5,label="predict_y5")
plt.scatter(y1["Log_Freq_HAL"], predict_std1_up,label="predict_std1_up")
plt.scatter(y1["Log_Freq_HAL"], predict_std1_low,label="predict_std1_low")
plt.scatter(y2["Log_Freq_HAL"], predict_std2_up,label="predict_std2_up")
plt.scatter(y2["Log_Freq_HAL"], predict_std2_low,label="predict_std2_low")
plt.scatter(y3["Log_Freq_HAL"], predict_std3_up,label="predict_std3_up")
plt.scatter(y3["Log_Freq_HAL"], predict_std3_low,label="predict_std3_low")
plt.scatter(y4["Log_Freq_HAL"], predict_std4_up,label="predict_std4_up")
plt.scatter(y4["Log_Freq_HAL"], predict_std4_low,label="predict_std4_low")
plt.scatter(y5["Log_Freq_HAL"], predict_std5_up,label="predict_std5_up")
plt.scatter(y5["Log_Freq_HAL"], predict_std5_low,label="predict_std5_low")




# plt.plot(xd, piecewise_linear(xd, *p))
plt.legend()
plt.xlabel("frequency")
plt.ylabel("response_time")
plt.savefig(data_path/"elp_freq_rt.png")
plt.close()