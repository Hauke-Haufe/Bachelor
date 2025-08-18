from pathlib import Path

from slam_test import SlamTest
from tsdf_test import TsdfTest


def main():
    print("===== Running TSDF Test =====")
    tsdf = TsdfTest()
    print("TSDF results saved to:", Path("data/odometry/testResults/tsdf_results.csv").resolve())

    print("\n===== Running SLAM Test =====")
    slam = SlamTest()
    print("SLAM results saved to:", Path("data/odometry/testResults/slam_results.csv").resolve())

    print("\nAll tests finished successfully!")


if __name__ == "__main__":
    main()