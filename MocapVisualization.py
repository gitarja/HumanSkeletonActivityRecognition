from pymo.parsers import BVHParser
from pymo.viz_tools import draw_stickfigure3d
from pymo.preprocessing import MocapParameterizer
import matplotlib.pyplot as plt

parser = BVHParser()

parsed_data = parser.parse('D:\\usr\\pras\\data\\HumanActivity\\Mocap\\SkeletalData\\skl_s01_a01_r01.bvh')

mp = MocapParameterizer('position')

positions = mp.fit_transform([parsed_data])

draw_stickfigure3d(positions[0], frame=120)

plt.show()
