import argparse
from pathlib import Path
import shutil

def read_file_list(filename):
    with open(filename) as file:
        lines = file.read().replace(",", " ").replace("\t", " ").split("\n")
    parsed = [[v.strip() for v in line.split(" ") if v.strip() != ""] for line in lines if len(line) > 0 and line[0] != "#"]
    return dict((float(l[0]), l[1:]) for l in parsed if len(l) > 1)

def associate_one_to_one(list1, list2, offset, max_difference):
    """
    Greedy 1-to-1 association based on closest timestamp within max_difference.
    Each timestamp is matched only once.
    """
    keys1 = set(list1.keys())
    keys2 = set(list2.keys())

    potential_matches = [
        (abs(a - (b + offset)), a, b)
        for a in keys1
        for b in keys2
        if abs(a - (b + offset)) < max_difference
    ]

    potential_matches.sort()
    matches = []
    for diff, a, b in potential_matches:
        if a in keys1 and b in keys2:
            matches.append((a, b))
            keys1.remove(a)
            keys2.remove(b)
    matches.sort()
    return matches

def find_nearest_groundtruth(timestamp, groundtruth_dict):
    """
    Find the groundtruth entry closest in time to the given timestamp.
    """
    best_key = min(groundtruth_dict.keys(), key=lambda k: abs(k - timestamp), default=None)
    return best_key

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Alignes RGB, Depth and Groundtruth timestamps')
    parser.add_argument('root_dir', help='RGB text file (format: timestamp filename)')
    parser.add_argument('--offset', type=float, default=0.0, help='Time offset for depth timestamps')
    parser.add_argument('--max_difference', type=float, default=0.02, help='Max allowed timestamp difference')
    args = parser.parse_args()

    root_dir = Path(args.root_dir)

    rgb_file = root_dir /"rgb.txt"
    depth_file = root_dir / "depth.txt"
    gt_file = root_dir / "groundtruth.txt"

    rgb_list = read_file_list(rgb_file)
    depth_list = read_file_list(depth_file)
    groundtruth_list = read_file_list(gt_file)

    matches = associate_one_to_one(rgb_list, depth_list, args.offset, args.max_difference)

    rgb_aligned_dir = root_dir/ (rgb_file.stem + "_aligned")
    depth_aligned_dir = root_dir/ (depth_file.stem + "_aligned")
    gt_aligned_file = root_dir / "groundtruth_aligned.txt"

    rgb_aligned_dir.mkdir(exist_ok=True)
    depth_aligned_dir.mkdir(exist_ok=True)

    with open(gt_aligned_file, "w") as gt_out:
        gt_out.write("# timestamp tx ty tz qx qy qz qw\n")
        for a, b in matches:
            rgb_src = root_dir/ rgb_list[a][0]
            depth_src = root_dir / depth_list[b][0]

            rgb_dst = rgb_aligned_dir / Path(rgb_list[a][0]).name
            depth_dst = depth_aligned_dir / Path(depth_list[b][0]).name

            shutil.copy(rgb_src, rgb_dst)
            shutil.copy(depth_src, depth_dst)

            gt_time = find_nearest_groundtruth(a, groundtruth_list)
            if gt_time is not None:
                gt_data = " ".join(groundtruth_list[gt_time])
                gt_out.write(f"{a:.6f} {gt_data}\n")

            print(f"{a:.6f} {rgb_list[a][0]} <-> {b:.6f} {depth_list[b][0]} | GT: {gt_time:.6f}")
