from slam import odometry_system
from icp import icp

class reconstructive_system():

    def __init__(self, odometry_backend):

        self.Odometry = odometry_system(odometry_backend)
        self.Icp = icp()

    def run_system(self):
        
        self.Odometry.run_system("data/images")
        self.Icp.run_system()




