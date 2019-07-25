import math
import numpy as np
class Person(object):
    '''
    an object representing a person whose has 15 joints, in which each joint consists of three axis and confidences [x, y, z, confidence_score]
    '''

    def __init__(self, id, prediction_score, joints, time):
        '''
        axis = {confidence,x, y}
        '''
        self.id = id
        self.prediction_score = prediction_score
        self.time = time

        self.joints = self.Joint(joints)


    def toJson(self):
        return dict(id=self.id, prediction_score=self.prediction_score, joints=self.joints, time=self.time)

    def getCoordinates(self):
        x = [self.joints.toJson()[joint].x for joint in self.joints.toJson()]
        y = [self.joints.toJson()[joint].y for joint in self.joints.toJson()]
        z = [self.joints.toJson()[joint].z for joint in self.joints.toJson()]

        return x, y, z

    def getSkeleton(self):
        x = [[pair[0].x, pair[1].x] for pair in self.joints.toSkeleton()]
        y = [[pair[0].y, pair[1].y] for pair in self.joints.toSkeleton()]
        z = [[pair[0].z, pair[1].z] for pair in self.joints.toSkeleton()]

        return x, y, z

    def getFlattenCoordinates(self):
        coordinate = [[self.joints.toJson()[joint].x, self.joints.toJson()[joint].y, self.joints.toJson()[joint].z] for joint in self.joints.toJson()]

        return np.array(coordinate).flatten()

    def setCoordinates(self, x, y, z):
        i = 0
        for joint in self.joints.toJson():
            self.joints.toJson()[joint].x = x[i]
            self.joints.toJson()[joint].y = y[i]
            self.joints.toJson()[joint].z = z[i]
            i+=1


    def getOrigin(self):
        return self.joints.getOrigin()








    class Joint(object):


        def __init__(self, joints):
            self.HEAD = self.Coordinate(joints["HEAD"])
            self.NECK = self.Coordinate(joints["NECK"])
            self.RIGHT_SHOULDER = self.Coordinate(joints["RIGHT_SHOULDER"])
            self.RIGHT_ELBOW = self.Coordinate(joints["RIGHT_ELBOW"])
            self.RIGHT_WRIST = self.Coordinate(joints["RIGHT_WRIST"])
            self.LEFT_SHOULDER = self.Coordinate(joints["LEFT_SHOULDER"])
            self.LEFT_ELBOW = self.Coordinate(joints["LEFT_ELBOW"])
            self.LEFT_WRIST = self.Coordinate(joints["LEFT_WRIST"])
            self.RIGHT_HIP = self.Coordinate(joints["RIGHT_HIP"])
            self.RIGHT_KNEE = self.Coordinate(joints["RIGHT_KNEE"])
            self.RIGHT_ANKLE = self.Coordinate(joints["RIGHT_ANKLE"])
            self.LEFT_HIP = self.Coordinate(joints["LEFT_HIP"])
            self.LEFT_KNEE = self.Coordinate(joints["LEFT_KNEE"])
            self.LEFT_ANKLE = self.Coordinate(joints["LEFT_ANKLE"])
            self.CHEST = self.Coordinate(joints["CHEST"])
            self.coordinateDist()




        def toJson(self):
            return dict(HEAD=self.HEAD,
                        NECK=self.NECK,
                        CHEST=self.CHEST,
                        RIGHT_SHOULDER=self.RIGHT_SHOULDER,
                        RIGHT_ELBOW=self.RIGHT_ELBOW,
                        RIGHT_WRIST=self.RIGHT_WRIST,
                        LEFT_SHOULDER=self.LEFT_SHOULDER,
                        LEFT_ELBOW=self.LEFT_ELBOW,
                        LEFT_WRIST=self.LEFT_WRIST,
                        RIGHT_HIP=self.RIGHT_HIP,
                        RIGHT_KNEE=self.RIGHT_KNEE,
                        RIGHT_ANKLE=self.RIGHT_ANKLE,
                        LEFT_HIP=self.LEFT_HIP,
                        LEFT_KNEE=self.LEFT_KNEE,
                        LEFT_ANKLE=self.LEFT_ANKLE,

                        )

        def toSkeleton(self):
            return [[self.HEAD, self.NECK], [self.NECK, self.RIGHT_SHOULDER], [self.RIGHT_SHOULDER, self.RIGHT_ELBOW], [self.RIGHT_ELBOW, self.RIGHT_WRIST],
                    [self.NECK, self.LEFT_SHOULDER], [self.LEFT_SHOULDER, self.LEFT_ELBOW], [self.LEFT_ELBOW, self.LEFT_WRIST], [self.CHEST, self.RIGHT_HIP],
                    [self.RIGHT_HIP, self.RIGHT_KNEE], [self.RIGHT_KNEE, self.RIGHT_ANKLE], [self.CHEST, self.LEFT_HIP], [self.LEFT_HIP, self.LEFT_KNEE],
                    [self.LEFT_KNEE, self.LEFT_ANKLE], [self.NECK, self.CHEST]]

        def normalize(self):
            self.HEAD.substract(pair=self.CHEST)
            self.NECK.substract(pair=self.CHEST)
            self.RIGHT_SHOULDER.substract(pair=self.CHEST)
            self.RIGHT_ELBOW.substract(pair=self.CHEST)
            self.RIGHT_WRIST.substract(pair=self.CHEST)
            self.LEFT_SHOULDER.substract(pair=self.CHEST)
            self.LEFT_ELBOW.substract(pair=self.CHEST)
            self.LEFT_WRIST.substract(pair=self.CHEST)
            self.RIGHT_HIP.substract(pair=self.CHEST)
            self.RIGHT_KNEE.substract(pair=self.CHEST)
            self.RIGHT_ANKLE.substract(pair=self.CHEST)
            self.LEFT_HIP.substract(pair=self.CHEST)
            self.LEFT_KNEE.substract(pair=self.CHEST)
            self.LEFT_ANKLE.substract(pair=self.CHEST)
            self.CHEST.substract(pair=self.CHEST)


        def coordinateDist(self):

            #compute distance between joint1_joint2

            #head anc chest
            self.HEAD_NECK = self.Coordinate(joints_axis=self.diffCoordinate(self.HEAD, self.NECK))
            self.NECK_CHEST = self.Coordinate(joints_axis=self.diffCoordinate(self.NECK, self.CHEST))

            #neck to left hand
            self.LEFT_WRIST_ELBOW = self.Coordinate(joints_axis=self.diffCoordinate(self.LEFT_WRIST, self.LEFT_ELBOW))
            self.LEFT_ELBOW_SHOULDER = self.Coordinate(joints_axis=self.diffCoordinate(self.LEFT_ELBOW, self.LEFT_SHOULDER))
            self.LEFT_SHOULDER_NECK = self.Coordinate(joints_axis=self.diffCoordinate(self.LEFT_SHOULDER, self.NECK))


            #neck to right hand
            self.RIGHT_WRIST_ELBOW = self.Coordinate(joints_axis=self.diffCoordinate(self.RIGHT_WRIST, self.RIGHT_ELBOW))
            self.RIGHT_ELBOW_SHOULDER = self.Coordinate(joints_axis=self.diffCoordinate(self.RIGHT_ELBOW, self.RIGHT_SHOULDER))
            self.RIGHT_SHOULDER_NECK = self.Coordinate(joints_axis=self.diffCoordinate(self.RIGHT_SHOULDER, self.NECK))

            #chest to left hip
            self.LEFT_ANKLE_KNEE = self.Coordinate(joints_axis=self.diffCoordinate(self.LEFT_ANKLE, self.LEFT_KNEE))
            self.LEFT_KNEE_HIP = self.Coordinate(joints_axis=self.diffCoordinate(self.LEFT_KNEE, self.LEFT_HIP))
            self.LEFT_HIP_CHEST = self.Coordinate(joints_axis=self.diffCoordinate(self.LEFT_HIP, self.CHEST))

            # chest to right hip
            self.RIGHT_ANKLE_KNEE = self.Coordinate(joints_axis=self.diffCoordinate(self.RIGHT_ANKLE, self.RIGHT_KNEE))
            self.RIGHT_KNEE_HIP = self.Coordinate(joints_axis=self.diffCoordinate(self.RIGHT_KNEE, self.RIGHT_HIP))
            self.RIGHT_HIP_CHEST = self.Coordinate(joints_axis=self.diffCoordinate(self.RIGHT_HIP, self.CHEST))




        def diffCoordinate(self, coordinate1, coordinate2):
            return self.sqrtDist(coordinate1.x - coordinate2.x), self.sqrtDist(coordinate1.y - coordinate2.y), self.sqrtDist(coordinate1.z - coordinate2.z)
            #return coordinate1.x - coordinate2.x, coordinate1.y - coordinate2.y, coordinate1.z - coordinate2.z

        def sqrtDist(self, x):
            return math.sqrt(math.pow(x, 2))


        def getOrigin(self):
            xOrigin = (self.CHEST.x + self.RIGHT_HIP.x + self.LEFT_HIP.x) / 3
            yOrigin = (self.CHEST.y + self.RIGHT_HIP.y + self.LEFT_HIP.y) / 3
            zOrigin = (self.CHEST.z + self.RIGHT_HIP.z + self.LEFT_HIP.z) / 3

            return self.Coordinate(joints_axis=[xOrigin, yOrigin, zOrigin])
        class Coordinate(object):
            x = 0
            y = 0
            z = 0
            confidence = 0

            def __init__(self, axis = None, joints_axis=None):
                if axis is not None:
                    self.confidence = axis["confidence"]
                    self.x = axis["x"]
                    self.y = axis["y"]
                    self.z = axis["z"]
                if joints_axis is not None:
                    self.x = joints_axis[0]
                    self.y = joints_axis[1]
                    self.z = joints_axis[2]

            def toJson(self):
                return dict(x=self.x, y=self.y, z=self.z, confidence=self.confidence)

            def toArray(self):
                return np.array([self.x, self.y, self.z, self.confidence])

            def toList(self):
                return [self.x, self.y, self.z, self.confidence]

            def substract(self, pair=None, list=None):
                if pair is not None and list is not None:
                    xOpt = 1
                    yOpt = 1
                    zOpt = 1
                    if self.x > pair.x:
                        xOpt = -1
                    if self.y > pair.y:
                        yOpt = -1
                    if self.z > pair.z:
                        zOpt = -1

                    self.x = self.x + (list[0]  * xOpt)
                    self.y = self.y + (list[1]  * yOpt)
                    self.z = self.z + (list[2]  * zOpt)

                    # self.x = self.x + list[0]
                    # self.y = self.y + list[1]
                    # self.z = self.z + list[2]
                elif pair is not None:
                    self.x = self.x - pair.x
                    self.y = self.y - pair.y
                    self.z = self.z - pair.z







