from pymo.parsers import BVHParser
import pymo.viz_tools
from pymo.preprocessing import Filtering
import yaml
import os
import pandas as pd


with open("conf/setting.yml") as ymfile:
    cfg = yaml.load(ymfile)

parser = BVHParser()

dir_mocap = cfg["DIR_MOCAP"]
dir_result = cfg["DIR_MOCAP_RESULT"]
dir_normalize_result = cfg["DIR_MOCAP_NORMALIZED_RESULT"]

filenames = os.listdir(dir_result)
data = pd.DataFrame(columns={"filename", "action"})
i = 0
for filename in filenames:
#     partsFilename= filename.split("_")
#     if len(partsFilename[2]) ==3:
#         data.loc[i] = [filename, int(partsFilename[2][1:])]
#         i+=1
    parsed_data = parser.parse(os.path.join(dir_result, filename))
    filter = Filtering()
    skeleton = filter.smoothing(parsed_data.values)
    skeleton = filter.normalize(skeleton)

    skeleton.to_pickle(os.path.join(dir_normalize_result, filename.split(".")[0] + ".pkl"))


#data.to_csv(os.path.join(dir_mocap, "dataset.csv"))

