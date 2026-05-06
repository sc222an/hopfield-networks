### analysis.py 
### Hopfield Network Energy and Capacity Analysis

Implements a Hopfield Neural Network to evaluate memory recall capacity and visualise energy landscapes using the Olivetti Faces dataset. 

**Components**
*   **Energy Visualisation**: Projects the high-dimensional energy landscape into 2D contour maps and smooth 3D surface plots using `tanh`, visualising memory attractors between target and stored faces.
*   **Outputs**: Generates grids of recalled faces, performance plots (Accuracy, Error, Energy, and Load Ratio vs. Stored Faces), and exports summary metrics to `hopfield_results.csv`.

### hopfieldfaces.py
### Hopfield Network Noise Analysis

Evaluates the robustness of a Hopfield Network against varying levels of input noise.

**Components**
*   **Noise Threshold Experiment**: Introduces incrementally higher levels of noise (10% to 60%) into the target cue on a fixed memory load of 6 faces.
*   **Outputs**: Plots grids combining original targets, corrupted cues, recalled patterns, and a performance curve to show recall accuracy against increasing noise levels.
