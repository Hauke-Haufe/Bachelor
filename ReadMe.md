# Project

Dieses Repository besteht aus zwei Hauptteilen:
1. **Image Segmentation**
2. **Odometry**

Für die Einrichtung des Grund-Repositories sowie Installation aller Abhängigkeiten:
```bash
python setup.py
```

# Dataset vorbereiten
Ersetze den dataset-Ordner durch den Datensatz. 

# Traning
Bevor ein Modelle traniert werden koennen müssen zuert die Folds erzeugt werden mit Train-Vaslidationsplits erstellt werden. 
```bash
python image_segmentation/scripts/create_folds.py <path/to/dataset> --k_folds <num_folds>
```
Um einen Hyperparamter Optierungstrial auf einen Fold laufen zu lassen kann folgender Command genutzt werden
```bash
CUDA_VISIBLE_DEVICES=<device_id> python image_segmentation/scripts/run_optimization.py <path/to/specific/fold> --num_iterations <num_iteration_optimizer>
```
Nach einem Optimerungsdurchlauf können das beste Modell per Fold Trainiert werden
```bash
CUDA_VISIBLE_DEVICES=<device_id> python image_segmentation/scripts/train_best.py <path/to/folds>
```



