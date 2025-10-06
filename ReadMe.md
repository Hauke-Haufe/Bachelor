# Project

Dieses Repository besteht aus zwei Bestandteilen:
1. **Image Segmentation**
2. **Odometry**

Für die Einrichtung des Grund-Repositories sowie Installation aller Abhängigkeiten:
```bash
python setup.py
```
# Inference
Um Inference auf einem Ordner von Bildern laufen zu lassen kann ein bereitgestelltes Script benutzt werden.
```bash
python image_segmentation/scripts/predict.py <path/to/modelcheckpoint> <path/to/image/folder> --save_overlays=true
```
Es wird ein Ordner in dem Zielordner erstellt, der Visualisierungen der Preictions enthält.

# Traning

## Dataset vorbereiten
Ersetze den dataset-Ordner durch den Datensatz. 

## Fold Erstellung
Bevor ein Modelle traniert werden koennen müssen zuert die Folds erzeugt werden mit Train-Vaslidationsplits erstellt werden. 
```bash
python image_segmentation/scripts/create_folds.py <path/to/dataset> --k_folds <num_folds>
```
Die Splits werden im dem Datensatz Ordner erstellt.
## Optimierung
Um einen Hyperparamter Optierungstrial auf einen Fold laufen zu lassen kann folgender Command genutzt werden
```bash
CUDA_VISIBLE_DEVICES=<device_id> python image_segmentation/scripts/run_optimization.py <path/to/specific/fold> --num_iterations <num_iteration_optimizer>
```
Nach einem Optimerungsdurchlauf kann das beste Modell per Fold trainiert werden.
```bash
CUDA_VISIBLE_DEVICES=<device_id> python image_segmentation/scripts/train_best.py <path/to/folds>
```

# Annotation 
Für die  Annotations Sofware siehe image_segmentation/labeln/LabelStudioWorkflow.py. Viel manuelles Setup ist benötigt!!!

