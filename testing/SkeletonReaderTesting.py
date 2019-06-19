import yaml
import os
from kinect.SkeletonReader import SkeletonReader
import pandas as pd
with open("../conf/setting.yml") as ymfile:
    cfg = yaml.load(ymfile)

dir = cfg["DIR_MSR"]
dir_msr = cfg["DIR_MSR_RESULT"]
dir_msr_normalized = cfg["DIR_MSR_NORMALIZED_RESULT"]
reader = SkeletonReader()


filenames = os.listdir(dir_msr)
data = pd.DataFrame(columns=["filename", "subject", "action"])
i = 0
for filename in filenames:
    filename_parts = filename.split("_")
    action = int(filename_parts[0][1:])
    subject = int(filename_parts[1][1:])

    data.loc[i] = [filename, subject, action]

    # skeleton = reader.read(os.path.join(dir_msr, filename))
    # skeleton.to_pickle(os.path.join(dir_msr_normalized, filename.split(".")[0]+".pkl"))
    i+=1


data.to_csv(os.path.join(dir, "dataset.csv"))
