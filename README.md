```bash
conda create -n ov_py310 python=3.10 -y
conda activate ov_py310


pip install -r requirements.txt
pip install openvino==2023.1.0.dev20230811 openvino-dev==2023.1.0.dev20230811 onnx onnxruntime
pip install -e .

mkdir weights
cd weights/
wget -q https://github.com/IDEA-Research/GroundingDINO/releases/download/v0.1.0-alpha/groundingdino_swint_ogc.pth
cd ..

```

# export

```bash
python demo/export_openvino.py -c groundingdino/config/GroundingDINO_SwinT_OGC.py -p weights/groundingdino_swint_ogc.pth -o weights/
```


# test

```bash
#pytorch
python demo/inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino_swint_ogc.pth \
-i .asset/demo7.jpg \
-t "Horse. Clouds. Grasses. Sky. Hill." \
-o logs/1111 \
 --cpu-only
```


```bash
#openvino
python demo/ov_inference_on_a_image.py \
-c groundingdino/config/GroundingDINO_SwinT_OGC.py \
-p weights/groundingdino.xml \
-i .asset/demo7.jpg  \
-t " Horse. Clouds. Grasses. Sky. Hill."  \
-o logs/2222 -d CPU
```

```bash
#numpy
```