# E9 robustness summary (synthetic degradation, ScanNet200 val subset)

- scenes: scene0025_00, scene0050_00, scene0063_00, scene0064_00, scene0164_00, scene0169_00, scene0193_00, scene0196_00, scene0207_00, scene0249_00
- protocol: 200-frame per-frame segmentation (--dense-seg), same as e5 baseline
- baseline 'none' = /home/OnlineAnySeg/e5_out_200_dense (clean e5 outputs, not re-run)

| condition | AP | AP50 | AP25 |
|---|---|---|---|
| none | 0.2143 | 0.3947 | 0.6170 |
| blur_w | 0.2120 | 0.3854 | 0.6087 |
| blur_s | 0.2257 | 0.4202 | 0.6315 |
| blur_xs | 0.2168 | 0.4069 | 0.5972 |
| noise_w | 0.2190 | 0.4026 | 0.6324 |
| noise_s | 0.2203 | 0.3987 | 0.6343 |
| noise_xs | 0.2148 | 0.3861 | 0.6171 |
| drop133 | 0.1928 | 0.3484 | 0.5443 |
| drop100 | 0.1696 | 0.3263 | 0.4996 |
| drop67 | 0.1323 | 0.2760 | 0.4290 |
