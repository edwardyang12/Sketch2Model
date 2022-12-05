#!/bin/sh
# Pixel2Style
cd /edward-slow-vol/Sketch2Model/pixel2style2pixel/
if [ "$1" = airplane ]; then
python scripts/inference.py --exp_dir=/edward-slow-vol/Sketch2Model/test/image --checkpoint_path=/edward-slow-vol/Sketch2Model/sketch2model/test1/checkpoints/best_model.pt --data_path=/edward-slow-vol/Sketch2Model/test/sketch/airplane --test_batch_size=1 --test_workers=4 --resize_outputs
else 
python scripts/inference.py --exp_dir=/edward-slow-vol/Sketch2Model/test/image --checkpoint_path=/edward-slow-vol/Sketch2Model/sketch2model/test2/checkpoints/best_model.pt --data_path=/edward-slow-vol/Sketch2Model/test/sketch/chair --test_batch_size=1 --test_workers=4 --resize_outputs
fi
mv /edward-slow-vol/Sketch2Model/test/image/inference_results/* /edward-slow-vol/Sketch2Model/test/image

# Pixel2Mesh
cd /edward-slow-vol/Sketch2Model/Pixel2Mesh/
python entrypoint_predict.py --checkpoint /edward-slow-vol/Sketch2Model/Pixel2Mesh/checkpoints/test/1128135218/090321_000033.pt --folder /edward-slow-vol/Sketch2Model/test/image --name model
