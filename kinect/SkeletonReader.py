import numpy as np
import pandas as pd


class SkeletonReader:
    columns = ["head_xposition", "head_yposition", "head_zposition",
               "neck_xposition", "neck_yposition", "neck_zposition",
               "chest_xposition", "chest_yposition", "chest_zposition",
               "pelvis_xposition", "pelvis_yposition", "pelvis_zposition",
               "l_shoulder_xposition", "l_shoulder_yposition", "l_shoulder_zposition",
               "l_elbow_xposition", "l_elbow_yposition", "l_elbow_zposition",
               "l_wrist_xposition", "l_wrist_yposition", "l_wrist_zposition",
               "l_arm_xposition", "l_arm_yposition", "l_arm_zposition",
               "r_shoulder_xposition", "r_shoulder_yposition", "r_shoulder_zposition",
               "r_elbow_xposition", "r_elbow_yposition", "r_elbow_zposition",
               "r_wrist_xposition", "r_wrist_yposition", "r_wrist_zposition",
               "r_arm_xposition", "r_arm_yposition", "r_arm_zposition",
               "l_hip_xposition", "l_hip_yposition", "l_hip_zposition",
               "l_kneel_xposition", "l_kneel_yposition", "l_kneel_zposition",
               "l_angkle_xposition", "l_angkle_yposition", "l_angkle_zposition",
               "l_toe_xposition", "l_toe_yposition", "l_toe_zposition",
               "r_hip_xposition", "r_hip_yposition", "r_hip_zposition",
               "r_kneel_xposition", "r_kneel_yposition", "r_kneel_zposition",
               "r_angkle_xposition", "r_angkle_yposition", "r_angkle_zposition",
               "r_toe_xposition", "r_toe_yposition", "r_toe_zposition"]

    def read(self, file):
        f = open(file, "r")
        lines = np.array(f.readlines())
        num_frame = int(lines[0].split(" ")[0])
        k = 0
        n=0
        data = pd.DataFrame(columns=self.columns)
        try:
            for i in range(0, num_frame):

                num_skel = int(lines[1 + (k * 41) + (n*40)])

                if num_skel != 0:
                    data_range = np.arange((2 + k) + (k * 40)+ (n*40), (2 + k) + ((k + 1) * 40)+ (n*40), 2)
                    #print(data_range)
                    joints = lines[data_range]

                    skeleton = self.convertSkeleton(joints)[:, 0:3]
                    origin = np.average(skeleton[[7, 5, 6], :], axis=0)
                    skeleton = skeleton - origin
                    trunk = skeleton[[19, 2, 3, 6], :]
                    l_arm = skeleton[[0, 7, 9, 11], :]
                    r_arm = skeleton[[1, 8, 10, 12], :]
                    l_leg = skeleton[[4, 13, 15, 17], :]
                    r_leg = skeleton[[5, 14, 16, 18], :]
                    data.loc[i] = np.concatenate(
                        [trunk.flatten(), l_arm.flatten(), r_arm.flatten(), l_leg.flatten(), r_leg.flatten()])
                    k += 1

                    if num_skel == 80:
                        n+=1
        except:
            print(file)

        return self.smoothing(data)


    def convertSkeleton(self, joints):
        converted_joint = []
        for joint in joints:
            converted_joint.append([float(j) for j in joint.split(" ")])

        return np.array(converted_joint)


    def smoothing(self, X):
        X = pd.concat([X[:2], X, X[-2:]], ignore_index=True)
        for i in range(2, len(X.index) - 2):
            X.loc[i] = (-3 * X.loc[i - 2] + 12 * X.loc[i - 1] + 17 * X.loc[i] + 12 * X.loc[i + 1] - 3 * X.loc[i + 2]) / 35

        return X[2:-2]
