## Odometry & TSDF Tests

Um die Odometry- und TSDF-Tests auszuführen, muss zunächst eine **angepasste Version von Open3D** kompiliert werden.  
Dies geschieht im Verzeichnis:
```
 odometry/lib/Open3D_fork
```
### Open3D kompilieren
Folge dem offiziellen Kompilations-Guide von Open3D:  
👉 [Open3D Compilation Guide](https://www.open3d.org/docs/release/compilation.html)

Es wird empfohlen, beim Kompilieren ein **Build Prefix** anzugeben. Mithilfe des `install`-Verzeichnisses kann anschließend das Projekt mit dem bereitgestellten Script kompiliert werden:

```bash
python odometry/compile.py --open3d_dir=<path/to/open3d_install/lib/cmake/Open3D>
```

# Test ausführen
1. **Platziere das TUM Dataset ind das Verzeichnis **data/odometry/TUMDataset** **
2. **Starte die Simulation mit**
```bash
python odometry/run_test.py
```
Die Resultate der Tests werden automatisch unter folgendem Pfad gespeichert: 
```
data/odometry/testResults
```