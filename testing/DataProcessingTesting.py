from utils.SkeletonGenerator import SkeletonGenerator
import yaml
import os
with open("../conf/setting.yml") as ymfile:
    cfg = yaml.load(ymfile)

dir_mocap = cfg["DIR_MOCAP"]
generator = SkeletonGenerator(batch_size=4, skeleton_path=cfg["DIR_MOCAP_NORMALIZED_RESULT"], dataset_path=os.path.join(dir_mocap, "train.csv"), t=20, n_class=11, train=True)
x, y = generator.getFlow(5)
