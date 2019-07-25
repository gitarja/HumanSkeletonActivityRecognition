import numpy as np


class SkeletonPreProcessing:

    def distance(self, points, means, std):
        return np.sqrt(np.power(points - means, 2)) > std

    def filter(self, x, dist_name=""):
        '''

        :param x: the distance of joints in the body
        :param joint: the name of joint that wants to be normalized
        - HEAD_NECK = 13
        - NECK_CHEST = 0
        - LEFT_WRIST_ELBOW = 6
        - LEFT_ELBOW_SHOULDER = 5
        - LEFT_SHOULDER_NECK = 1
        - RIGHT_WRIST_ELBOW = 8
        - RIGHT_ELBOW_SHOULDER = 7
        - RIGHT_SHOULDER_NECK = 2
        - LEFT_ANKLE_KNEE = 10
        - LEFT_KNEE_HIP = 9
        - LEFT_HIP_CHEST = 4
        - RIGHT_ANKLE_KNEE = 12
        - RIGHT_KNEE_HIP = 11
        - RIGHT_HIP_CHEST = 3





        :return: the normalized joint
        '''
        y = x.copy()
        if dist_name == 13:
            means_abs = np.array([0.0709333, 0.1297303, 0.13748])
            std = np.array([0.06333017, 0.05741198, 0.03690964])
            # means_ori = np.array([-0.02089844, -0.02831209, 0.11713041])
        elif dist_name == 0:
            means_abs = np.array([0.0209259, 0.0123695, 0.28623])
            std = np.array([0.02181384, 0.01659776, 0.0370593])
            # means_ori = np.array([0.0048221, 0.01118659, 0.27677735])
        elif dist_name == 6:
            # from data
            means_abs = np.array([0.06000062, 0.0598235, 0.185945])
            std = np.array([0.18250654, 0.1962549, 0.10923468])
            # means_abs = np.array([0.14241475, 0.14241475, 0.14241475])
            # std = np.array([0.028383575, 0.028383575, 0.028383575])
            # means_ori = np.array([-0.01083734, -0.02139975, -0.08048818])
        elif dist_name == 5:
            # from data
            means_abs = np.array([0.0634575, 0.090513, 0.190285])
            std = np.array([0.12334926, 0.12409572, 0.10380413])
            # means_abs = np.array([0.1368418915, 0.08391482, 0.1368418915])
            # std = np.array([0.034472429, 0.06938827, 0.034472429])
            # means_ori = np.array([0.02555016, 0.00311051, -0.24095594])
        elif dist_name == 1:
            # from data
            means_abs = np.array([0.0832065, 0.07892215, 0.00918])
            std = np.array([0.04037403, 0.04862166, 0.01883761])
            # means_abs = np.array([0.119439409, 0.06384462, 0.0189468])
            # std = np.array([0.042100574, 0.0506868, 0.02528662])
            # means_ori = np.array([0.01769817, -0.01756307, -0.01123141])
        elif dist_name == 8:
            # from data
            means_abs = np.array([0.1025841, 0.0539935, 0.1681015])
            std = np.array([0.16955857, 0.18749399, 0.10847652])
            # means_abs = np.array([0.14241475, 0.14241475, 0.14241475])
            # std = np.array([0.028383575, 0.028383575, 0.028383575])
            # means_ori = np.array([-0.00997583, -0.00292648, -0.06836219])
        elif dist_name == 7:
            # from data
            means_abs = np.array([0.070718, 0.0904354, 0.20269])
            std = np.array([0.12423768, 0.12143614, 0.09967501])
            # means_abs = np.array([0.1368418915, 0.08391482, 0.1368418915])
            # std = np.array([0.034472429, 0.06938827, 0.034472429])
            # means_ori = np.array([-0.00560513, 0.0137177, -0.24444452])
        elif dist_name == 2:
            # from data
            means_abs = np.array([0.09384205, 0.07996, 0.01343])
            std = np.array([0.06363088, 0.05620045, 0.0234423])
            # means_abs = np.array([0.119439409, 0.06384462, 0.0189468])
            # std = np.array([0.042100574, 0.0506868, 0.02528662])
            # means_ori = np.array([-0.02388771, 0.02280688, -0.00549265])
        elif dist_name == 10:
            # from data
            means_abs = np.array([0.033967, 0.0384205, 0.3781859])
            std = np.array([0.05021612, 0.04070084, 0.04358558])
            # means_abs = np.array([0.255340981, 0.255340981, 0.255340981])
            # std = np.array([0.051338709, 0.051338709, 0.051338709])
            # means_ori = np.array([-0.00141776, 0.02959571, -0.33565366])
        elif dist_name == 9:
            # from data
            means_abs = np.array([0.031704, 0.0282905, 0.40322275])
            std = np.array([0.04069311, 0.04187992, 0.04392536])
            # means_abs = np.array([0.142095383, 0.09429914, 0.38824083])
            # std = np.array([0.08241516, 0.08979575, 0.12181161])
            # means_ori = np.array([-0.01126459, -0.04711003, -0.36502925])

        elif dist_name == 4:
            # form data
            means_abs = np.array([0.050043, 0.0478495, 0.1759965])
            std = np.array([0.02794277, 0.03648264, 0.02072678])
            # means_abs = np.array([0.140104784, 0.05043462, 0.140104784])
            # std = np.array([0.015822131, 0.05062104, 0.015822131])
            # means_ori = np.array([0.00925347, -0.00411409, -0.162824])
        elif dist_name == 12:
            # from data
            means_abs = np.array([0.0398876, 0.033333, 0.384959])
            std = np.array([0.04051596, 0.03763689, 0.04594542])
            # means_abs = np.array([0.255340981, 0.255340981, 0.255340981])
            # std = np.array([0.051338709, 0.051338709, 0.051338709])
            # means_ori = np.array([0.01469589, 0.00575057, -0.3479777])
        elif dist_name == 11:
            # from data
            means_abs = np.array([0.03739145, 0.0260875, 0.4179485])
            std = np.array([0.03971686, 0.04048913, 0.03830229])
            # means_abs = np.array([0.142095383, 0.09429914, 0.38824083])
            # std = np.array([0.08241516, 0.08979575, 0.12181161])
            # means_ori = np.array([-0.02992455, -0.02554857, -0.36677003])
        elif dist_name == 3:
            # from data
            means_abs = np.array([0.052141, 0.05600335, 0.1799975])
            std = np.array([0.03226508, 0.03509221, 0.02203046])
            # means_abs = np.array([0.140104784, 0.05043462, 0.140104784])
            # std = np.array([0.015822131, 0.05062104, 0.015822131])
            # means_ori = np.array([-0.0134982, 0.01810158, -0.15468776])

        # for i in range(3, len(x) - 3, 1):
        #     # y[i] = (6 * y[i - 1, :] + 1 * y[i, :] + 6 * y[i + 1, :]) / 13
        #     y[i] = (-3 * y[i - 3, :] + 6 * y[i - 2, :] + 6 * y[i - 2, :] + 17 * y[i, :] + 6 * y[i + 1, :] + 6 * y[i + 2,
        #                                                                                                         :] - 3 * y[
        #                                                                                                                  i + 3,
        #                                                                                                                  :]) / 35
        # return (-1 * (y - means_ori) *  self.distance(y_abs, means_abs, std))
        # return (-1 * (y - means_abs)) *  (y_abs > (means_abs + std))
        return (y - (means_abs + std)) * (y > (means_abs + std))

    def smoothing(self, joints, padding = 2):
        """"to get smoothed skeleton for ith frame do smoothing by considering
            joint[0] = joint_i-1
            joint[1] = joint_i-2
            joint[2] = joint_i
            joint[3] = joint_i+1
            joint[4] = joint_i+2            
        """""
        # smoothed_join = []
        # for i in range(2, len(joints) - 2, 1):
        #     if (joints[i - 2] == 0):
        #         f0 = [0 * j for j in joints[i]]
        #     else:
        #         f0 = [-3 / 35 * j for j in joints[i - 2]]
        #
        #     if (joints[i - 1] == 0):
        #         f1 = [0 * j for j in joints[i]]
        #     else:
        #         f1 = [12 / 35 * j for j in joints[i - 1]]
        #
        #     f2 = [17 / 35 * j for j in joints[i]]
        #
        #     if (joints[i + 1] == 0):
        #         f3 = [0 * j for j in joints[i]]
        #     else:
        #         f3 = [12 / 35 * j for j in joints[i + 1]]
        #
        #     if (joints[i + 2] == 0):
        #         f4 = [0 * j for j in joints[i]]
        #     else:
        #         f4 = [-3 / 35 * j for j in joints[i + 2]]
        #
        #     f = [a0 + a1 + a2 + a3 + a4 for a0, a1, a2, a3, a4 in zip(f0, f1, f2, f3, f4)]
        #
        #     smoothed_join.append(f)
        #
        # return smoothed_join
        joints = np.append(np.insert(joints, 0, np.zeros((padding, len(joints[0]))), axis=0), np.zeros((padding, len(joints[0]))), axis=0)
        y = joints.copy()
        for i in range(padding, len(joints) - padding, 1):
            y[i] = (-3 * y[i - 2, :] + 12 * y[i-1, :] + 17 * y[i, :] + 12 * y[i+1, :] -3 * y[i+2, :]) / 35

        return y[padding:-padding]

    def preProcessing(self, persons):
        joints_x = []
        joints_y = []
        joints_z = []
        for index, row in persons.iterrows():
            x, y, z = row["person"].getCoordinates()
            joints_x.append(x)
            joints_y.append(y)
            joints_z.append(z)
        # print(persons.loc[1]["person"].getCoordinates())
        joints_x = self.smoothing([0, 0] + joints_x + [0, 0])
        joints_y = self.smoothing([0, 0] + joints_y + [0, 0])
        joints_z = self.smoothing([0, 0] + joints_z + [0, 0])

        for index, row in persons.iterrows():
            row["person"].setCoordinates(joints_x[index], joints_y[index], joints_z[index])

        return persons

    def filtering(self, person):
        distances = np.zeros((1, 3))
        for i in range(14):

            if i == 13:
                distances[0, 0] = person.joints.HEAD_NECK.x
                distances[0, 1] = person.joints.HEAD_NECK.y
                distances[0, 2] = person.joints.HEAD_NECK.z
            elif i == 0:
                distances[:, 0] = person.joints.NECK_CHEST.x
                distances[:, 1] = person.joints.NECK_CHEST.y
                distances[:, 2] = person.joints.NECK_CHEST.z
            elif i == 6:
                distances[:, 0] = person.joints.LEFT_WRIST_ELBOW.x
                distances[:, 1] = person.joints.LEFT_WRIST_ELBOW.y
                distances[:, 2] = person.joints.LEFT_WRIST_ELBOW.z
            elif i == 5:
                distances[:, 0] = person.joints.LEFT_ELBOW_SHOULDER.x
                distances[:, 1] = person.joints.LEFT_ELBOW_SHOULDER.y
                distances[:, 2] = person.joints.LEFT_ELBOW_SHOULDER.z
            elif i == 1:
                distances[:, 0] = person.joints.LEFT_SHOULDER_NECK.x
                distances[:, 1] = person.joints.LEFT_SHOULDER_NECK.y
                distances[:, 2] = person.joints.LEFT_SHOULDER_NECK.z
            elif i == 8:
                distances[:, 0] = person.joints.RIGHT_WRIST_ELBOW.x
                distances[:, 1] = person.joints.RIGHT_WRIST_ELBOW.y
                distances[:, 2] = person.joints.RIGHT_WRIST_ELBOW.z
            elif i == 7:
                distances[:, 0] = person.joints.RIGHT_ELBOW_SHOULDER.x
                distances[:, 1] = person.joints.RIGHT_ELBOW_SHOULDER.y
                distances[:, 2] = person.joints.RIGHT_ELBOW_SHOULDER.z
            elif i == 2:
                distances[:, 0] = person.joints.RIGHT_SHOULDER_NECK.x
                distances[:, 1] = person.joints.RIGHT_SHOULDER_NECK.y
                distances[:, 2] = person.joints.RIGHT_SHOULDER_NECK.z
            elif i == 10:
                distances[:, 0] = person.joints.LEFT_ANKLE_KNEE.x
                distances[:, 1] = person.joints.LEFT_ANKLE_KNEE.y
                distances[:, 2] = person.joints.LEFT_ANKLE_KNEE.z
            elif i == 9:
                distances[:, 0] = person.joints.LEFT_KNEE_HIP.x
                distances[:, 1] = person.joints.LEFT_KNEE_HIP.y
                distances[:, 2] = person.joints.LEFT_KNEE_HIP.z
            elif i == 4:
                distances[:, 0] = person.joints.LEFT_HIP_CHEST.x
                distances[:, 1] = person.joints.LEFT_HIP_CHEST.y
                distances[:, 2] = person.joints.LEFT_HIP_CHEST.z
            elif i == 12:
                distances[:, 0] = person.joints.RIGHT_ANKLE_KNEE.x
                distances[:, 1] = person.joints.RIGHT_ANKLE_KNEE.y
                distances[:, 2] = person.joints.RIGHT_ANKLE_KNEE.z
            elif i == 11:
                distances[:, 0] = person.joints.RIGHT_KNEE_HIP.x
                distances[:, 1] = person.joints.RIGHT_KNEE_HIP.y
                distances[:, 2] = person.joints.RIGHT_KNEE_HIP.z
            elif i == 3:
                distances[:, 0] = person.joints.RIGHT_HIP_CHEST.x
                distances[:, 1] = person.joints.RIGHT_HIP_CHEST.y
                distances[:, 2] = person.joints.RIGHT_HIP_CHEST.z

            residue = self.filter(distances, i)
            j = 0

            if i == 13:
                person.joints.HEAD.substract(pair=person.joints.NECK, list=residue[j, :])
            elif i == 0:
                person.joints.NECK.substract(pair=person.joints.CHEST, list=residue[j, :])
            elif i == 6:
                person.joints.LEFT_WRIST.substract(pair=person.joints.LEFT_ELBOW, list=residue[j, :])

            elif i == 5:
                person.joints.LEFT_ELBOW.substract(pair=person.joints.LEFT_SHOULDER, list=residue[j, :])

            elif i == 1:
                person.joints.LEFT_SHOULDER.substract(pair=person.joints.NECK, list=residue[j, :])

            elif i == 8:
                person.joints.RIGHT_WRIST.substract(pair=person.joints.RIGHT_ELBOW, list=residue[j, :])

            elif i == 7:
                person.joints.RIGHT_ELBOW.substract(pair=person.joints.RIGHT_SHOULDER, list=residue[j, :])

            elif i == 2:
                person.joints.RIGHT_SHOULDER.substract(pair=person.joints.NECK, list=residue[j, :])

            elif i == 10:
                person.joints.LEFT_ANKLE.substract(pair=person.joints.LEFT_KNEE, list=residue[j, :])

            elif i == 9:
                person.joints.LEFT_KNEE.substract(pair=person.joints.LEFT_HIP, list=residue[j, :])

            # elif i == 4:
            #     person.joints.LEFT_HIP.substract(pair=person.joints.CHEST, list=residue[j, :])

            elif i == 12:
                person.joints.RIGHT_ANKLE.substract(pair=person.joints.RIGHT_KNEE, list=residue[j, :])

            elif i == 11:
                person.joints.RIGHT_KNEE.substract(pair=person.joints.RIGHT_HIP, list=residue[j, :])

            # elif i == 3:
            #     person.joints.RIGHT_HIP.substract(pair=person.joints.CHEST, list=residue[j, :])

            j += 1
            person.joints.coordinateDist()

        return person
