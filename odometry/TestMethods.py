import json
import subprocess
from itertools import product
from pathlib import Path
import pandas as pd

def fmt(x):
    return "-" if x is None or pd.isna(x) else f"{x:.4f}"

class OdometryTest:

    def __init__(self):
        self.methods = ["Intensity", "Hybrid", "P2P"]
        self.scenes = ["static_xyz", "static_rpy", "walking_static", "walking_xyz", "walking_rpy"]
        self.masking = ["NoMask","Both"]

        self.methods_t_lookup = {
            "Intensity": "Intensität",
            "Hybrid": "Hybrid",
            "P2P": "P2P"
        }

        self.scene_t_lookup = {
            "static_xyz" : "static xyz", 
            "static_rpy" : "static rpy", 
            "walking_static": "walking static", 
            "walking_xyz": "walking xyz", 
            "walking_rpy": "walking rpy"
        }

        self.masking_t_lookup = {
            "Both": "maskiert", 
            "NoMask": ""
        }

    def compute_metrics(self):
        
        results = []
        for method, scene, mask in product(self.methods, self.scenes, self.masking):

            with open("data/testConfigs/odometry_config.json", "w") as f:
                json.dump({"device": "CUDA", 
                        "method": method,
                        "maskingmethod":mask, 
                        "dataset": scene}, f, indent=4)
                
            run_res = subprocess.run(["./executables/OdometryTest"])
        
            with open("data/output.json", "r") as f:
                return_v = json.load(f)
                return_v["method"] = method
                return_v["mask_method"] = mask
                return_v["scene"] = scene

                results.append(return_v)

            with open("data/odom_results.json", "w") as f:
                json.dump(results, f, indent = 4)

    def compute_time(self):
        pass

    def build_tab_rot_trans(self, scenes, lines):

        header_scene = (
            "    \\multirow{2}{*}{\\textbf{Methode}} & " +
            " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{self.scene_t_lookup[scene]}}}}}" 
                       for scene in scenes) +" \\\\"
        )

        header_sub = (
            "    & " +
            " & ".join("\\textbf{Trans. [m]} & \\textbf{Rot. [rad]}" for _ in scenes) +
            " \\\\"
        )

        latex = "\\begin{tabular}{l" + "*{"+str(len(scenes)*2)+"}{c}}" + "\n"
        latex += "\\toprule\n"
        latex += header_scene + "\n"
        latex += "\\cmidrule(lr){" + "}  \\cmidrule(lr){".join(
            [f"{i*2+2}-{i*2+3}" for i in range(len(scenes))]
        ) + "}\n"
        latex += header_sub + "\n"
        latex += "\\midrule\n"
        latex += "\n".join(lines) + "\n"
        latex += "\\bottomrule\n\\end{tabular}\n"

        return latex

    def metrics_to_latex(self, scenes, methods, metric):

        with open("data/testResults/odom_results.json", "r") as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        df = df.round(4)

         
        lines = []
        for method in methods:
            for masking in self.masking:
                label = f"{self.methods_t_lookup[method]} {self.masking_t_lookup[masking]}"
                row_parts = [label]
                for scene in scenes:
                    vals = df[
                        (df["method"] == method) &
                        (df["mask_method"] == masking) &
                        (df["scene"] == scene)
                    ]
                    if not vals.empty:
                        trans = fmt(vals.iloc[0][f"{metric}_Trans"])
                        rot = fmt(vals.iloc[0][f"{metric}_Rot"])
                    else:
                        trans, rot = "-", "-"
                    row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        # Assemble LaTeX table

        latex = self.build_tab_rot_trans(scenes, lines)

        Path(f"data/testResults/odometry_{metric}.tex").write_text(latex, encoding="utf-8")

    def rel_improvment_mask_to_latex(self, scenes, methods, metric):

        with open("data/testResults/odom_results.json", "r") as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        df = df.round(4)

        lines = []
        for method in methods:
            label = f"{self.methods_t_lookup[method]}"
            row_parts = [label]
            for scene in scenes:

                un_masked = df[
                    (df["method"] == method) &
                    (df["mask_method"] == "NoMask") &
                    (df["scene"] == scene)
                ]

                masked = df[
                    (df["method"] == method) &
                    (df["mask_method"] == "Both") &
                    (df["scene"] == scene)
                ]

                if not un_masked.empty and masked.empty:
                    trans = un_masked.iloc[0][f"{metric}_Trans"] - masked.iloc[0][f"{metric}_Trans"]
                    trans = fmt(trans /un_masked.iloc[0][f"{metric}_Trans"])
                    rot = un_masked.iloc[0][f"{metric}_Rot"] - masked.iloc[0][f"{metric}_Rot"]
                    rot = fmt(rot /un_masked.iloc[0][f"{metric}_Rot"])
                else:
                    trans, rot = "-", "-"

                row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_tab_rot_trans(scenes, lines)

        Path(f"data/testResults/odometry_rel_{metric}.tex").write_text(latex, encoding="utf-8")

class SlamTest:
    
    def __init__(self):
        self.methods = ["Intensity", "Hybrid", "P2P"]
        self.scenes = ["static_xyz", "static_rpy", "walking_static", "walking_xyz", "walking_rpy"]
        self.masking = ["Raw", "Maskout"]

        self.methods_t_lookup = {
            "Intensity": "Intensität",
            "Hybrid": "Hybrid",
            "P2P": "P2P"
        }

        self.scene_t_lookup = {
            "static_xyz" : "static xyz", 
            "static_rpy" : "static rpy", 
            "walking_static": "walking static", 
            "walking_xyz": "walking xyz", 
            "walking_rpy": "walking rpy"
        }

        self.masking_t_lookup = {
            "Raw": "", 
            "Maskout": "maskiert"
        }

    def compute_metrics(self):
    
        results = []
        for method, scene, mask in product(self.methods, self.scenes, self.masking):

            with open("data/testConfigs/slam_config.json", "w") as f:
                json.dump({"device": "CUDA", 
                        "method": method,
                        "slam_method":mask, 
                        "dataset": scene}, f, indent=4) #runs on the cpu so tracking loss doesnt explode vram
            try:
                r = subprocess.run(["./executables/SlamTest"], capture_output=True, text=True, check=True)
                success = True

            except subprocess.CalledProcessError as e:  
                success = False

            if success:
                with open("data/output.json", "r") as f:
                    return_v = json.load(f)
                    return_v["method"] = method
                    return_v["slam_method"] = mask
                    return_v["scene"] = scene

                    results.append(return_v)
            else:
                return_v = {
                    "ATE_Rot": None,
                    "ATE_Trans": None,
                    "RPE_Rot": None,
                    "RPE_Trans":None
                }
                return_v["method"] = method
                return_v["slam_method"] = mask
                return_v["scene"] = scene
                results.append(return_v)

            with open("data/testResults/slam_results.json", "w") as f:
                json.dump(results, f, indent = 4)

        with open("data/testResults/slam_results.json", "w") as f:
            json.dump(results,f, indent = 4)

    def compute_time(self):
        pass

    def build_tab_rot_trans(self, scenes, lines):

        header_scene = (
            "    \\multirow{2}{*}{\\textbf{Methode}} & " +
            " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{self.scene_t_lookup[scene]}}}}}" 
                       for scene in scenes) +" \\\\"
        )

        header_sub = (
            "    & " +
            " & ".join("\\textbf{Trans. [m]} & \\textbf{Rot. [rad]}" for _ in scenes) +
            " \\\\"
        )

        latex = "\\begin{tabular}{l" + "*{"+str(len(scenes)*2)+"}{c}}" + "\n"
        latex += "\\toprule\n"
        latex += header_scene + "\n"
        latex += "\\cmidrule(lr){" + "}  \\cmidrule(lr){".join(
            [f"{i*2+2}-{i*2+3}" for i in range(len(scenes))]
        ) + "}\n"
        latex += header_sub + "\n"
        latex += "\\midrule\n"
        latex += "\n".join(lines) + "\n"
        latex += "\\bottomrule\n\\end{tabular}\n"

        return latex
    
    def metrics_to_latex(self, scenes, methods, metric):

        with open("data/testResults/slam_results.json", "r") as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        df = df.round(4)

        lines = []
        for method in methods:
            for masking in self.masking:
                label = f"{self.methods_t_lookup[method]} {self.masking_t_lookup[masking]}"
                row_parts = [label]
                for scene in scenes:
                    vals = df[
                        (df["method"] == method) &
                        (df["slam_method"] == masking) &
                        (df["scene"] == scene)
                    ]
                    if not vals.empty:
                        trans = fmt(vals.iloc[0][f"{metric}_Trans"])
                        rot = fmt(vals.iloc[0][f"{metric}_Rot"])
                    else:
                        trans, rot = "-", "-"
                    row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_latex_rot_trans(scenes, lines)

        Path(f"data/testResults/tsdf_{metric}.tex").write_text(latex, encoding="utf-8")

    def rel_improvment_mask_to_latex(self, scenes, methods, metric):

        with open("data/testResults/slam_results.json", "r") as f:
            results = json.load(f)

        df = pd.DataFrame(results)
        df = df.round(4)

        lines = []
        for method in methods:
            label = f"{self.methods_t_lookup[method]}"
            row_parts = [label]
            for scene in scenes:

                un_masked = df[
                    (df["method"] == method) &
                    (df["mask_method"] == "Raw") &
                    (df["scene"] == scene)
                ]

                masked = df[
                    (df["method"] == method) &
                    (df["mask_method"] == "Maskout") &
                    (df["scene"] == scene)
                ]

                if not un_masked.empty and masked.empty:
                    trans = un_masked.iloc[0][f"{metric}_Trans"] - masked.iloc[0][f"{metric}_Trans"]
                    trans = fmt(trans /un_masked.iloc[0][f"{metric}_Trans"])
                    rot = un_masked.iloc[0][f"{metric}_Rot"] - masked.iloc[0][f"{metric}_Rot"]
                    rot = fmt(rot /un_masked.iloc[0][f"{metric}_Rot"])
                else:
                    trans, rot = "-", "-"

                row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_tab_rot_trans(scenes, lines)

        Path(f"data/testResults/tsdf_rel_{metric}.tex").write_text(latex, encoding="utf-8")

if __name__ == "__main__":

    test = OdometryTest()
    test.rel_improvment_mask_to_latex(["static_xyz", "static_rpy", "walking_static", "walking_xyz", "walking_rpy"],
                  ["Intensity", "Hybrid", "P2P"], "ATE")
    