#!/bin/bash
pip install lmdb pandas scikit-image trimesh shapely easydict && apt install unzip

# Pixel2Style
unzip /edward-slow-vol/Sketch2Model/stylegan2-pytorch/ninja-linux.zip -d /usr/local/bin/
update-alternatives --install /usr/bin/ninja ninja /usr/local/bin/ninja 1 --force

# Pixel2Mesh
cd /edward-slow-vol/Sketch2Model/Pixel2Mesh/external/chamfer/
python setup.py install
cd /edward-slow-vol/Sketch2Model/Pixel2Mesh/external/neural_renderer/
python setup.py install
