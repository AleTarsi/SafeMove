# SafeMove
SafeMove is an open-source project that aims to showcase how AI can be used to simplify ergonomic risk assessment through the REBA standard.

|Prediction|Analysis|
|---|---|
|<img src="./_assets/test_image.png" width=700;/>|<img src="./_assets/reba_example.png" width=700;/>|

## Dependencies:
The algorithm depends on [MediaPipe](https://github.com/google-ai-edge/mediapipe) a suite of AI tools to perform different visual tasks, such as human pose landmark estimation.

1. Install `pip`
    ```bash
    python -m ensure pip --default-pip
    ```
2. Install the following packages:
    ```bash
    pip install xlsxwriter pandas mediapipe opencv-python
    ```

## Usage:
Enter in the project directory and launch the following script:

```bash
python scripts/sm_01_main.py
```

## Results:
Enter in the folder `output` to visualize the results of your test.

You can visualize risk as body part graphs or as automatic generated excel containing the NN predictions over time.

|Body Parts|Excel|
|---|---|
|<img src="./_assets/angle_risk_example.png" wodth=600;/>|<img src="./_assets/excel_example.png" wodth=600;/>|

## Licence:

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


