from scene_fragmenter import Scene_fragmenter
from icp import icp

class reconstructive_system():

    def __init__(self, odometry_backend):

        self.Odometry = Scene_fragmenter(odometry_backend)
        self.Icp = icp()

    def run_system(self):
        
        self.Odometry.run_system("data/images")
        self.Icp.run_system()




