import json
import subprocess
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math
from matplotlib.patches import Patch

def fmt(x, r=4):
    return "\\textbf{--}" if x is None or pd.isna(x) else f"{x:.{r}f}"

class Test:

    def __init__(self):

        self.f_functions = ["Intensity", "Hybrid", "P2P"]
        self.scenes = ["static_xyz", "static_rpy", "walking_static", "walking_xyz", "walking_rpy"]
        self.devices = ["CUDA", "CPU"]

        self.f_functions_t_lookup = {
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
        
    def build_tab_m_rad(self, scenes, lines):

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
    
    def build_tab_pcm_prad(self, scenes, lines):
        header_scene = (
            "    \\multirow{2}{*}{\\textbf{Methode}} & " +
            " & ".join(f"\\multicolumn{{2}}{{c}}{{\\textbf{{{self.scene_t_lookup[scene]}}}}}" 
                       for scene in scenes) +" \\\\"
        )

        header_sub = (
            "    & " +
            " & ".join("\\textbf{Trans. [cm\\%]} & \\textbf{Rot. [rad\\%]}" for _ in scenes) +
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

    def check_input(self, scenes, methods, metric):

        if not set(scenes) <= set(self.scenes):
            raise RuntimeError("invalid Scenes")

        if not set(methods) <= set(self.f_functions):
            raise RuntimeError("invalid Odometry Error Functions")
        
        if not(metric == "RPE" or  metric == "ATE"):
            raise RuntimeError("invalid metric")

    def mertics_to_latex_var(self,scenes, err_function, metric): 

        lines = []
        for f in err_function:
            for masking in self.method:
                label = f"{self.f_functions_t_lookup[f]} {self.method_t_lookup[masking]}"
                row_parts = [label]
                for scene in scenes:
                    vals = self.df[
                        (self.df["err_function"] == f) &
                        (self.df["method"] == masking) &
                        (self.df["scene"] == scene)
                    ]
                    if not vals.empty:

                        trans = fmt(vals.iloc[0][f"{metric}_Trans"]) 
                        rot = fmt(vals.iloc[0][f"{metric}_Rot"]) 

                        devp_trans =  math.sqrt(vals.iloc[0][f"{metric}_Trans_var"])/ vals.iloc[0][f"{metric}_Trans"] *100
                        devp_rot =  math.sqrt(vals.iloc[0][f"{metric}_Rot_var"])/ vals.iloc[0][f"{metric}_Rot"] *100

                        devp_trans ="\pm " + fmt(devp_trans, 2)+ "\\%"
                        devp_rot ="\pm " + fmt(devp_rot, 2)+ "\\%"

                        trans =  "$" +trans + devp_trans +"$" 
                        rot = "$" + rot + devp_rot + "$" 
                    else:
                        trans, rot = "\\textbf{--}", "\\textbf{--}"
                    row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_tab_m_rad(scenes, lines)

        Path(f"data/odometry/testResults/{self.name}_{metric}_var.tex").write_text(latex, encoding="utf-8")

    def metrics_to_latex(self, scenes, err_function, metric):

        lines = []
        for f in err_function:
            for masking in self.method:
                label = f"{self.f_functions_t_lookup[f]} {self.method_t_lookup[masking]}"
                row_parts = [label]
                for scene in scenes:
                    vals = self.df[
                        (self.df["err_function"] == f) &
                        (self.df["method"] == masking) &
                        (self.df["scene"] == scene)
                    ]
                    if not vals.empty:
                        trans = fmt(vals.iloc[0][f"{metric}_Trans"])
                        rot = fmt(vals.iloc[0][f"{metric}_Rot"])
                    else:
                        trans, rot = "-", "-"
                    row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_tab_m_rad(scenes, lines)

        Path(f"data/odometry/testResults/{self.name}_{metric}.tex").write_text(latex, encoding="utf-8")

    def plot_metric_per_scene(self, scenes, err_functions, metric):

        x = np.arange(len(scenes)) 
        width = 0.13
        fig, ax = plt.subplots(1,2, figsize=(16, 4))
        fig.subplots_adjust(
            left=0.1,  
            right=0.9,  
            top=1.3,    
            bottom=0.1, 
            wspace=0.3, 
            hspace=0.3  
        )


        num_groups = len(err_functions)*2
        cmap = cm.get_cmap("Set3", num_groups)

        err = ["Trans", "Rot"]
        err_to_e ={"Trans": "m", "Rot": "rad"}

        j = 0
        for e in err:
            i = 0
            for f in err_functions:
                for mask in self.method:
                    means =[]
                    devs = []
                    for scene in scenes:

                        row= self.df[
                            (self.df["err_function"] == f) &
                            (self.df["method"] == mask) &
                            (self.df["scene"] == scene)
                        ]

                        if not row.empty:
                            mean = row.iloc[0][f"{metric}_{e}"]
                            var = row.iloc[0][f"{metric}_{e}_var"]

                        else:
                            mean, var = 0,0

                        means.append(mean)
                        devs.append(var)
                
                    offset = (i - num_groups / 2) * width + width / 2
                    ax[j].bar(x +offset, means, width, yerr = devs,  
                        label=f"{self.f_functions_t_lookup[f]} {self.method_t_lookup[mask]}", color=cmap(i))
                    i += 1
            

            ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].set_ylabel(err_to_e[e])
            ax[j].set_title(f" {e} {metric}")
            ax[j].set_xticks(x)
            ax[j].set_xticklabels(scenes)
            j+=1

        handles, labels = ax[0].get_legend_handles_labels()
        leg = fig.legend(
            handles, labels,
            loc="lower center",     
            ncol=3,                 
            frameon=False,           
            bbox_to_anchor=(0.5, -0.1)
        )

        fig.savefig(
            f"data/odometry/testResults/{self.name}_{metric}_com.png",
            dpi=300,
            bbox_inches='tight',
            bbox_extra_artists=(leg,)
        )
        plt.show()

    def plot_avg_time(self, scenes ,err_functions):

        x = np.arange(len(scenes)) 
        width = 0.13
        fig, ax = plt.subplots(1,2, figsize=(14, 3))
        fig.subplots_adjust(
            left=0.1,  
            right=0.9,  
            top=1.3,    
            bottom=0.1, 
            wspace=0.3, 
            hspace=0.3  
        )
        
        num_groups = 2
        cmap = cm.get_cmap("Set2", num_groups) 
        width = 0.3
        j = 0

        for scene in scenes:
            x = np.arange(len(err_functions)) 
            
            i = 0
            for mask in self.method:
                bars = []
                dev_bars =[]
                for f in err_functions:

                    row= self.df[
                        (self.df["err_function"] == f) &
                        (self.df["method"] == mask) &
                        (self.df["scene"] == scene)
                    ]

                    if not row.empty:
                        time = row.iloc[0]["avg_time"] /1000
                        time_dev = math.sqrt(row.iloc[0]["avg_time_var"])/ 1000
                    else:
                        time = 0
                    
                    bars.append(time)
                    dev_bars.append(time_dev)
                
                offset = (i - num_groups / 2) * width + width / 2
                ax[j].bar(x +offset, bars, width, label=self.method_t_lookup[mask], 
                    capsize=5, yerr = dev_bars, color=cmap(i))
                i += 1
                
            ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].set_ylabel("Zeit (ms)")
            ax[j].set_title(f"Durschnittliche Laufzeit - Scene {scene}")
            ax[j].set_xticks(x)
            ax[j].set_xticklabels(err_functions)
            ax[j].set_ylim(0, 24)
            j+=1

        handles, labels = ax[0].get_legend_handles_labels()
        leg = fig.legend(
            handles, labels,
            loc="lower center",      
            ncol=2,                  
            frameon=False,           
            bbox_to_anchor=(0.5, -0.1)
        )

        fig.savefig(
            f"data/odometry/testResults/{self.name}_time_avg.png",
            dpi=300,
            bbox_inches='tight',
            bbox_extra_artists=(leg,)
        )
        plt.show()


class OdometryTest(Test):

    def __init__(self):
        super().__init__()

        self.metrics = ["ATE_Trans", "ATE_Rot", "RPE_Trans", "RPE_Rot", 
                        "total_time", "avg_time"]
        
        self.name = "odom"

        self.method = ["NoMask","Both"]
        self.method_t_lookup = {
            "Both": "maskiert", 
            "NoMask": "default"
        }

        if not Path("data/odometry/testResults/odom_results.csv").is_file():
            self.compute_metrics()
    
        self.df = pd.read_csv("data/odometry/testResults/odom_results.csv")

    def compute_metrics(self):
        
        results = []
        for err, scene, mask in product(self.f_functions, self.scenes, self.method):

            values = {m: [] for m in self.metrics}
            for i in range(5):
                with open("data/odometry/testConfigs/odometry_config.json", "w") as f:
                    json.dump({"device": "CUDA", 
                            "method": err,
                            "maskingmethod":mask, 
                            "dataset": scene}, f, indent=4)
                    
                run_res = subprocess.run(["./executables/OdometryTest"])

                with open("data/output.json", "r") as f:
                    data = json.load(f)
                    for metric in self.metrics:
                        values[metric].append(data[metric])

            run_data = {}
            for metric in self.metrics:
                arr = np.array(values[metric], dtype=float)
                run_data[metric] = float(np.mean(arr)) if arr.size > 0 else None
                run_data[metric+ "_var"] = float(np.var(arr)) if arr.size > 0 else None
            
            run_data["err_function"] = err
            run_data["method"] = mask
            run_data["scene"] = scene
            results.append(run_data)

            df = pd.DataFrame(results)
            df.to_csv("data/odometry/testResults/odom_results.csv", index=False)

    def rel_improvment_mask_to_latex(self, scenes, methods, metric):

        self.check_input(scenes, methods, metric)

        lines = []
        for method in methods:
            label = f"{self.f_functions_t_lookup[method]}"
            row_parts = [label]
            for scene in scenes:

                un_masked = self.df[
                    (self.df["err_function"] == method) &
                    (self.df["method"] == "NoMask") &
                    (self.df["scene"] == scene)
                ]

                masked = self.df[
                    (self.df["err_function"] == method) &
                    (self.df["method"] == "Both") &
                    (self.df["scene"] == scene)
                ]

                if not un_masked.empty and masked.empty:
                    trans = un_masked.iloc[0][f"{metric}_Trans"] - masked.iloc[0][f"{metric}_Trans"]
                    trans = fmt(100*(trans /un_masked.iloc[0][f"{metric}_Trans"]))
                    rot = un_masked.iloc[0][f"{metric}_Rot"] - masked.iloc[0][f"{metric}_Rot"]
                    rot = fmt(100*(rot /un_masked.iloc[0][f"{metric}_Rot"]))
                else:
                    trans, rot = "-", "-"

                row_parts.extend([trans, rot])
                lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_tab_pcm_prad(scenes, lines)

        Path(f"data/odometry/testResults/odometry_rel_{metric}.tex").write_text(latex, encoding="utf-8")

    def plot_rel_improvment(self, scenes, methods, metric):

        self.check_input(scenes, methods, metric)

        x = np.arange(len(scenes)) 
        width = 0.25 
        fig, ax = plt.subplots(1,2, figsize=(10, 4))
        fig.subplots_adjust(
            left=0.1,   
            right=0.9,  
            top=0.9,    
            bottom=0.1, 
            wspace=0.3, 
            hspace=0.3  
        )

        num_groups = len(methods)
        cmap = cm.get_cmap("Set2", num_groups)
    
        err = ["Trans", "Rot"]
        
        j = 0
        for e in err:
            i = 0
            for method in methods:
                bars = []
                for scene in scenes:
                    un_masked = self.df[
                        (self.df["err_function"] == method) &
                        (self.df["method"] == "NoMask") &
                        (self.df["scene"] == scene)
                    ]

                    masked = self.df[
                        (self.df["err_function"] == method) &
                        (self.df["method"] == "Both") &
                        (self.df["scene"] == scene)
                    ]

                    if not un_masked.empty and not masked.empty:
                        trans = un_masked.iloc[0][f"{metric}_{e}"] - masked.iloc[0][f"{metric}_{e}"]
                        trans = 100*(trans /un_masked.iloc[0][f"{metric}_{e}"])
                    else:
                        trans,  0,0
                    
                    bars.append(trans)

                offset = (i - num_groups / 2) * width + width / 2
                ax[j].bar(x +offset, bars, width, label=method, color=cmap(i))
                i += 1

            ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].set_ylabel("%")
            ax[j].set_title(f"{e} {metric} Verbesserung")
            ax[j].set_xticks(x)
            ax[j].set_xticklabels(scenes)
            j+= 1

        fig.tight_layout()
        handles, labels = ax[0].get_legend_handles_labels()
        leg = fig.legend(
            handles, labels,
            loc="lower center",      # position
            ncol=3,                  # number of columns
            frameon=False,           # no box
            bbox_to_anchor=(0.5, -0.1)
        )

        fig.savefig(
            "data/odometry/testResults/odom_rel_imp.png",
            dpi=300,
            bbox_inches='tight',
            bbox_extra_artists=(leg,)
        )
        plt.show()

class SlamTest(Test):
    
    def __init__(self):
        super().__init__()
        self.metrics = ["ATE_Trans", "ATE_Rot", "RPE_Trans", "RPE_Rot", 
                        "total_time", "avg_time","avg_integration_time","avg_tracking_time", "avg_raycast_time"]
        
        self.method = ["Raw", "Maskout"]
        self.name = "tsdf"

        self.method_t_lookup = {
            "Raw": "default", 
            "Maskout": "maskiert"
        }

        if not Path("data/odometry/testResults/slam_results.csv").is_file():
            self.compute_metrics()
        
        self.df = pd.read_csv("data/odometry/testResults/slam_results.csv")

    def compute_metrics(self):
        
        results = []
        num_iterations = 5
        for err, scene, mask in product(self.f_functions, self.scenes, self.method):
            
            values = {m: [] for m in self.metrics}

            for i in range(num_iterations):

                with open("data/odometry/testConfigs/slam_config.json", "w") as f:
                    json.dump({"device": "CUDA", 
                            "method": err,
                            "slam_method":mask, 
                            "dataset": scene}, f, indent=4) #runs on the cpu so tracking loss doesnt explode vram
                try:
                    r = subprocess.run(["./executables/SlamTest"], capture_output=True, text=True, check=True)
                    success = True

                except subprocess.CalledProcessError as e:  
                    success = False

                if success:
                    with open("data/output.json", "r") as f:
                        data = json.load(f)

                        for metric in self.metrics:
                            values[metric].append(data[metric])
                else:
                    pass

            run_data = {}
            for metric in self.metrics:
                arr = np.array(values[metric], dtype=float).flatten()
                run_data[metric] = float(np.mean(arr,axis=0)) if arr.size == num_iterations else None
                run_data[metric+ "_var"] = float(np.var(arr)) if arr.size == num_iterations else None
            
            run_data["err_function"] = err
            run_data["method"] = mask
            run_data["scene"] = scene
            results.append(run_data)

            df = pd.DataFrame(results)
            df.to_csv("data/odometry/testResults/slam_results.csv", index=False)

    def rel_improvment_to_latex(self, scenes, err_function, metric):

        self.check_input(scenes, err_function, metric)

        lines = []
        for f in err_function:
            label = f"{self.f_functions_t_lookup[f]}"
            row_parts = [label]
            for scene in scenes:

                un_masked =  self.df[
                    ( self.df["err_function"] == f) &
                    ( self.df["method"] == "Raw") &
                    ( self.df["scene"] == scene)
                ]

                masked = self.df[
                    (self.df["err_function"] == f) &
                    (self.df["slam_method"] == "Maskout") &
                    (self.df["scene"] == scene)
                ]

                if not un_masked.empty and not masked.empty:
                    trans = un_masked.iloc[0][f"{metric}_Trans"] - masked.iloc[0][f"{metric}_Trans"]
                    trans = fmt(100*(trans /(un_masked.iloc[0][f"{metric}_Trans"])))
                    rot = un_masked.iloc[0][f"{metric}_Rot"] - masked.iloc[0][f"{metric}_Rot"]
                    rot = fmt(100*(rot /un_masked.iloc[0][f"{metric}_Rot"]))
                else:
                    trans, rot = "-", "-"

                row_parts.extend([trans, rot])
            lines.append(" & ".join(row_parts) + r" \\")

        latex = self.build_tab_pcm_prad(scenes, lines)

        Path(f"data/odometry/testResults/tsdf_rel_{metric}.tex").write_text(latex, encoding="utf-8")

    def plot_rel_improvment(self, scenes, err_function, metric):

        self.check_input(scenes, err_function, metric)

        x = np.arange(len(scenes)) 
        width = 0.25 
        fig, ax = plt.subplots(1,2, figsize=(10, 4))
        fig.subplots_adjust(
            left=0.1,   
            right=0.9,  
            top=0.9,    
            bottom=0.1, 
            wspace=0.3, 
            hspace=0.3  
        )

        num_groups = len(err_function)
        cmap = cm.get_cmap("Set2", num_groups)
    
        err = ["Trans", "Rot"]
        
        j = 0
        for e in err:
            i = 0
            for method in err_function:
                bars = []
                for scene in scenes:
                    un_masked = self.df[
                        (self.df["err_function"] == method) &
                        (self.df["method"] == "Raw") &
                        (self.df["scene"] == scene)
                    ]

                    masked = self.df[
                        (self.df["err_function"] == method) &
                        (self.df["method"] == "Maskout") &
                        (self.df["scene"] == scene)
                    ]

                    if not un_masked.empty and not masked.empty:
                        trans = un_masked.iloc[0][f"{metric}_{e}"] - masked.iloc[0][f"{metric}_{e}"]
                        trans = 100*(trans /un_masked.iloc[0][f"{metric}_{e}"])
                    else:
                        trans,  0,0
                    
                    bars.append(trans)

                offset = (i - num_groups / 2) * width + width / 2
                ax[j].bar(x +offset, bars, width, label=method, color=cmap(i))
                i += 1

            ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].set_ylabel("%")
            ax[j].set_title(f"{e} {metric} Verbesserung")
            ax[j].set_xticks(x)
            ax[j].set_xticklabels(scenes)
            j+= 1

        fig.tight_layout()
        handles, labels = ax[0].get_legend_handles_labels()
        leg = fig.legend(
            handles, labels,
            loc="lower center",      # position
            ncol=3,                  # number of columns
            frameon=False,           # no box
            bbox_to_anchor=(0.5, -0.1)
        )

        fig.savefig(
            "data/odometry/testResults/tsdf_rel_imp.png",
            dpi=300,
            bbox_inches='tight',
            bbox_extra_artists=(leg,)
        )
        plt.show()

    def plot_avg_time_detail(self, scenes, err_functions):

        x_scenes = np.arange(len(scenes))
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        fig.subplots_adjust(left=0.1, right=0.9, top=1.0, bottom=0.22, wspace=0.3, hspace=0.3)

        num_methods = len(self.method)
        cmap = cm.get_cmap("Set2", num_methods)
        width = 0.3

        for j, scene in enumerate(scenes):
            x = np.arange(len(err_functions))  # positions within this subplot

            for i, method in enumerate(self.method):
                bars_int, bars_trk, bars_ryc = [], [], []
                err_int, err_trk , err_ryc = [], [], []

                for f in err_functions:
                    row = self.df[
                        (self.df["err_function"] == f) &
                        (self.df["method"] == method) &
                        (self.df["scene"] == scene)
                    ]

                    if row.empty:
                        bars_int.append(0.0); bars_trk.append(0.0)
                        err_int.append(0.0);  err_trk.append(0.0)
                    else:
                        r0 = row.iloc[0]

                        time_ryc_ms = float(r0["avg_raycast_time"])  /1000
                        time_trk_ms = float(r0["avg_tracking_time"])  /1000    
                        time_int_ms = float(r0["avg_integration_time"])  /1000
                         
                        std_int_ms = float(r0.get("avg_integration_time_var", 0.0)) ** 0.5 /1000
                        std_trk_ms = float(r0.get("avg_tracking_time_var", 0.0)) ** 0.5 /1000
                        std_ryc_ms = float(r0.get("avg_raycast_time_var", 0.0)) ** 0.5 /1000

                        bars_int.append(time_int_ms)
                        bars_trk.append(time_trk_ms)
                        bars_ryc.append(time_ryc_ms)
                        err_int.append(std_int_ms)
                        err_trk.append(std_trk_ms)
                        err_ryc.append(std_ryc_ms)

                offset = (i - num_methods / 2) * width + width / 2

                ax[j].bar(
                    x + offset, bars_ryc, width,
                    label=f"raycast {self.method_t_lookup[method]}" if j == 0 else "",
                     capsize=6,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                    color=cmap(i), alpha=1, hatch="//", edgecolor="black", zorder=1
                )

                bottom_rt = (np.array(bars_ryc)).tolist()
                ax[j].bar(
                    x + offset, bars_trk, width,
                    label=f"tracking {self.method_t_lookup[method]}" if j == 0 else "",
                    bottom=bottom_rt,
                     capsize=6,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                    color=cmap(i), alpha=1, edgecolor="black", zorder=2
                )

                bottom_rti = (np.array(bars_ryc) + np.array(bars_trk)).tolist()
                ax[j].bar(
                    x + offset, bars_int, width,
                    label=f"integration {self.method_t_lookup[method]}" if j == 0 else "",
                    bottom=bottom_rti,
                     capsize=6,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                    color=cmap(i), alpha=0.6, edgecolor="black", zorder=3
                )

            ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
            ax[j].set_ylabel("Zeit (ms)")
            ax[j].set_title(f"Durchschnittliche Laufzeit – Scene {scene}")
            ax[j].set_xticks(x)
            ax[j].set_xticklabels(err_functions, rotation=0)
            ax[j].set_ylim(0, 27)
        

        handels, labels = ax[0].get_legend_handles_labels()

        leg_segments = fig.legend(
            handels, labels,
            loc="lower center", ncol=2, frameon=False, bbox_to_anchor=(0.5, -0.03)
        )

        # Save without cropping legends
        fig.savefig(
            f"data/odometry/testResults/{self.name}_time_avg_split.png",
            dpi=300, bbox_inches='tight'
        )
        plt.show()


def plot_drift_comparision(scenes, err_functions, metric):
    
    odom = OdometryTest()
    slam = SlamTest()

    x = np.arange(len(scenes)) 
    width = 0.13
    fig, ax = plt.subplots(1,2, figsize=(16, 4))
    fig.subplots_adjust(
        left=0.1,  
        right=0.9,  
        top=1.3,    
        bottom=0.1, 
        wspace=0.3, 
        hspace=0.3  
    )


    num_groups = len(err_functions)*2
    cmap = cm.get_cmap("Set3", num_groups)

    err = ["Trans", "Rot"]
    err_to_e ={"Trans": "m", "Rot": "rad"}

    identify = {
        "Both": "Maskout", 
        "NoMask": "Raw"
    }

    j = 0
    for e in err:
        i = 0
        for f in err_functions:
    
            for mask in odom.method:
                odom_means, slam_means =[], []
                odom_devs, slam_devs =[], []    
                for scene in scenes:

                    odom_row= odom.df[
                        (odom.df["err_function"] == f) &
                        (odom.df["method"] == mask) &
                        (odom.df["scene"] == scene)
                    ]
                    slam_row= slam.df[
                        (slam.df["err_function"] == f) &
                        (slam.df["method"] == identify[mask]) &
                        (slam.df["scene"] == scene)
                    ]

                    if not odom_row.empty and not slam_row.empty:
                        odom_mean = odom_row.iloc[0][f"{metric}_{e}"]
                        odom_var = odom_row.iloc[0][f"{metric}_{e}_var"] ** 0.5
                        slam_mean = slam_row.iloc[0][f"{metric}_{e}"]
                        slam_var = slam_row.iloc[0][f"{metric}_{e}_var"] ** 0.5

                    else:
                        odom_mean ,odom_var, slam_mean, slam_var = 0, 0, 0, 0

                    slam_means.append(slam_mean)
                    slam_devs.append(slam_var)

                    odom_means.append(odom_mean)
                    odom_devs.append(odom_var)
            
                offset = (i - num_groups / 2) * width + width / 2
                ax[j].bar(x +offset, odom_means, width, 
                    label=f"{odom.f_functions_t_lookup[f]} {odom.method_t_lookup[mask]}",
                    capsize=6,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                    color=cmap(i), alpha=0.5,  edgecolor="black")
                
                ax[j].bar(x +offset, slam_means, width, 
                    label=f"{slam.f_functions_t_lookup[f]} {slam.method_t_lookup[identify[mask]]}",
                     capsize=6,
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                    color=cmap(i), alpha=1, edgecolor="black")

                i += 1
        
        ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
        ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
        ax[j].set_ylabel(err_to_e[e])
        ax[j].set_title(f" {e} {metric}")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(scenes)
        j+=1

    handles, labels = ax[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="lower center",     
        ncol=3,                  
        frameon=False,           
        bbox_to_anchor=(0.5, -0.3)
    )

    fig.savefig(
        f"data/odometry/testResults/compose_{metric}_com.png",
        dpi=300,
        bbox_inches='tight',
        bbox_extra_artists=(leg,)
    )
    plt.show()

def plot_com_method(scenes, err_functions, metric, method):
    
    odom = OdometryTest()
    slam = SlamTest()

    x = np.arange(len(scenes)) 
    width = len(scenes)
    fig, ax = plt.subplots(1,2, figsize=(16, 4))
    fig.subplots_adjust(
        left=0.1,  
        right=0.9,  
        top=1.3,    
        bottom=0.1, 
        wspace=0.3, 
        hspace=0.3  
    )


    num_groups = len(err_functions)
    cmap = cm.get_cmap("Set3", num_groups)

    err = ["Trans", "Rot"]
    err_to_e ={"Trans": "m", "Rot": "rad"}

    identify = {
        "Both": "Maskout", 
        "NoMask": "Raw"
    }

    j = 0
    for e in err:
        i = 0
        for f in err_functions:

            odom_means, slam_means =[], []
            odom_devs, slam_devs =[], []    
            for scene in scenes:

                odom_row= odom.df[
                    (odom.df["err_function"] == f) &
                    (odom.df["method"] == method) &
                    (odom.df["scene"] == scene)
                ]
                slam_row= slam.df[
                    (slam.df["err_function"] == f) &
                    (slam.df["method"] == identify[method]) &
                    (slam.df["scene"] == scene)
                ]

                if not odom_row.empty and not slam_row.empty:
                    odom_mean = odom_row.iloc[0][f"{metric}_{e}"]
                    odom_var = odom_row.iloc[0][f"{metric}_{e}_var"] ** 0.5
                    slam_mean = slam_row.iloc[0][f"{metric}_{e}"]
                    slam_var = slam_row.iloc[0][f"{metric}_{e}_var"] ** 0.5

                else:
                    odom_mean ,odom_var, slam_mean, slam_var = 0, 0, 0, 0

                slam_means.append(slam_mean)
                slam_devs.append(slam_var)

                odom_means.append(odom_mean)
                odom_devs.append(odom_var)
            
                offset = (i - num_groups / 2) * width + width / 2
            ax[j].bar(x +offset, odom_means, width, 
                label=f"{odom.f_functions_t_lookup[f]} {odom.method_t_lookup[method]}",
                capsize=6,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                color=cmap(i), alpha=0.5,  edgecolor="black")
            
            ax[j].bar(x +offset, slam_means, width, 
                label=f"{slam.f_functions_t_lookup[f]} {slam.method_t_lookup[identify[method]]}",
                    capsize=6,
                error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                color=cmap(i), alpha=1, edgecolor="black")

            i += 1
        
        ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
        ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
        ax[j].set_ylabel(err_to_e[e])
        ax[j].set_title(f" {e} {metric}")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(scenes)
        j+=1

    handles, labels = ax[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="lower center",     
        ncol=3,                  
        frameon=False,           
        bbox_to_anchor=(0.5, -0.3)
    )

    fig.savefig(
        f"data/odometry/testResults/compose_{metric}_com.png",
        dpi=300,
        bbox_inches='tight',
        bbox_extra_artists=(leg,)
    )
    plt.show()

def plot_rel_com(scenes, err_functions, metric):
    
    odom = OdometryTest()
    slam = SlamTest()

    x = np.arange(len(scenes)) 
    width = 0.35
    fig, ax = plt.subplots(1,2, figsize=(16, 4))
    fig.subplots_adjust(
        left=0.1,  
        right=0.9,  
        top=1.3,    
        bottom=0.1, 
        wspace=0.3, 
        hspace=0.3  
    )


    num_groups = len(err_functions)*2
    cmap = cm.get_cmap("Set2", num_groups)

    err = ["Trans", "Rot"]
    err_to_e ={"Trans": "m", "Rot": "rad"}

    identify = {
        "Both": "Maskout", 
        "NoMask": "Raw"
    }

    j = 0
    for e in err:
        i = 0
        for f in err_functions:
    
            for mask in odom.method:
                percentage =[]
                for scene in scenes:

                    odom_row= odom.df[
                        (odom.df["err_function"] == f) &
                        (odom.df["method"] == mask) &
                        (odom.df["scene"] == scene)
                    ]
                    slam_row= slam.df[
                        (slam.df["err_function"] == f) &
                        (slam.df["method"] == identify[mask]) &
                        (slam.df["scene"] == scene)
                    ]

                    if not odom_row.empty and not slam_row.empty:
                        odom_mean = odom_row.iloc[0][f"{metric}_{e}"]
                        slam_mean = slam_row.iloc[0][f"{metric}_{e}"]

                        if math.isnan(slam_mean):
                            improv = 0
                        else:
                            improv = ((odom_mean-slam_mean) /odom_mean) *100

                    
                    else:
                        improv = 0

                    percentage.append(improv)

                offset = (i - num_groups / 2) * width + width / 2
                ax[j].bar(x +offset, percentage, width, 
                    label=f"{odom.f_functions_t_lookup[f]} {odom.method_t_lookup[mask]}",
                    error_kw={'elinewidth': 1.5, 'ecolor': 'black'},
                    color=cmap(i),  edgecolor="black")
                
                i += 1
        
        ax[j].yaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
        ax[j].xaxis.grid(True, linestyle="--", which="major", color="gray", alpha=0.7)
        ax[j].set_ylabel( "Relative Verbesserung (%)")
        ax[j].set_title(f" {e} {metric}")
        ax[j].set_xticks(x)
        ax[j].set_xticklabels(scenes)
        j+=1

    handles, labels = ax[0].get_legend_handles_labels()
    leg = fig.legend(
        handles, labels,
        loc="lower center",     
        ncol=4,                  
        frameon=False,           
        bbox_to_anchor=(0.5, -0.05)
    )

    fig.savefig(
        f"data/odometry/testResults/compose_{metric}_com.png",
        dpi=300,
        bbox_inches='tight',
        bbox_extra_artists=(leg,)
    )
    plt.show()

if __name__ == "__main__":
    
    test = OdometryTest()
    test.plot_avg_time(["static_xyz", "walking_xyz"], ["Intensity", "Hybrid", "P2P"])

    