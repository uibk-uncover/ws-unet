

```bash
pip install -r requirements.txt
```

Run the following commands from the `src/` subdirectory.

```bash
# prediction error
python filters/evaluate.py
python unet/evaluate.py

# plot error boxes per KB filter error
python error_boxes.py
# plot contours
python contour.py

# WS error
python ws/evaluate.py

# detection error - TODO
python ws/roc.py
python detector/evaluate.py

# correlation - TODO
python correlation.py

# saliency maps - TODO
python saliency.py
```
