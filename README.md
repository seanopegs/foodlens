# FoodLens

Food image recognition on Food-101. Final project for COMP6826001 Deep Learning, BINUS University (Group 4).

FoodLens takes a photo of a dish, tells you which of 101 dishes it is, and estimates its calories, protein, carbs and fat. We compare a CNN trained from scratch with two ImageNet-pretrained models (MobileNetV2 and EfficientNet-B0) on the same data split, then serve the chosen model in a small Streamlit web app.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/seanopegs/foodlens/blob/main/FoodLens_Experiments.ipynb)

## Team

| Name | Student ID |
|---|---|
| Sean Jeremy Henry Sihombing | 2802548530 |
| Josea Immanuel Dihardja | 2802543984 |
| Rheinald Hubert Zhang | 2802445936 |

## Repository

```
FoodLens_Experiments.ipynb   data exploration, training runs, results, error analysis
foodlens/data.py             loading, train/val/test split, preprocessing, DataLoaders
foodlens/models.py           Custom CNN, MobileNetV2, EfficientNet-B0
foodlens/engine.py           training loop, early stopping, metrics, timing, saving runs
foodlens/nutrition.py        nutrition estimate: portion size and extras
foodlens/nutrition.json      nutrients per typical serving of each dish (generated)
nutrition/servings.csv       Food-101 class -> USDA FNDDS food(s) and serving weight
nutrition/addons.csv         extras the user can add (oil, butter, cheese, egg, ...)
nutrition/drinks.csv         drinks the user can add to the meal
nutrition/build_nutrition.py builds nutrition.json from the FNDDS tables
app.py                       Streamlit web app
run_app.bat                  starts the app on Windows (installs the packages on first run)
.streamlit/config.toml       app theme
```

## Nutrition estimates

Every dish is mapped to the closest food(s) in the USDA Food and Nutrient Database for Dietary Studies (FNDDS 2021–2023) together with the weight of a typical serving, for example ramen → *Ramen bowl with meat and egg*, 3 cups (735 g). Dishes FNDDS does not list as a whole are built from their parts (fish and chips = fried fish + french fries). The `match` column in `nutrition/servings.csv` marks each mapping as `exact`, `closest` or `composite`.

In the app the user can correct the dish if the prediction is wrong, choose a smaller or larger portion (×0.7 / ×1.4), add extras such as oil in the broth, an egg or extra rice, and add a drink (Food-101 has no drink classes, so drinks are picked from a list in `nutrition/drinks.csv`); anything else can be entered as extra calories. The numbers are estimates for a typical serving, not a measurement of the photo.

To rebuild `nutrition.json`, download the FNDDS CSV release from [FoodData Central](https://fdc.nal.usda.gov/download-datasets) (we used `FoodData_Central_survey_food_csv_2024-10-31`) and run:

```bash
python nutrition/build_nutrition.py path/to/FoodData_Central_survey_food_csv_2024-10-31
```

## Dataset

[Food-101](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) (Bossard et al., 2014), loaded from Hugging Face as [`ethz/food101`](https://huggingface.co/datasets/ethz/food101): 101,000 RGB photos, 101 classes, 1,000 per class.

| Split | Images | Per class | Source |
|---|---:|---:|---|
| Train | 68,175 | 675 | official train split, 90% (stratified, seed 42) |
| Validation | 7,575 | 75 | official train split, 10% (stratified, seed 42) |
| Test | 25,250 | 250 | official test split (named `validation` on Hugging Face) |

Preprocessing: resize the shorter side to 256 px, crop 224 × 224 (random for training, centre for evaluation), ImageNet mean/std normalisation. Training images also get a horizontal flip, ±15° rotation and colour jitter.

## Experiments

Every scenario changes one variable from the baseline and uses the same split. Shared settings: Adam, cross-entropy loss, up to 15 epochs with early stopping on validation loss (patience 3), seed 42.

| ID | Variable changed | Value |
|---|---|---|
| Baseline | | Custom CNN, learning rate 1e-3, batch size 32, augmentation on |
| E1 | learning rate | 1e-4 |
| E2 | batch size | 16 |
| E3 | architecture | MobileNetV2, ImageNet weights, last 3 feature blocks + new head trained |
| E4 | architecture | EfficientNet-B0, ImageNet weights, last 3 feature blocks + new head trained |

Metrics: accuracy, top-5 accuracy, macro precision / recall / F1 on the test split, plus training time, parameter count and single-image inference time.

## Results

Trained on Kaggle (Tesla T4 GPU), one *Save & Run All* version, 4.4 hours for all five runs. Test split: 25,250 photos.

| Run | Model | Test accuracy | Top-5 | Macro F1 | Best / last epoch | Training time | CPU time per photo |
|---|---|---:|---:|---:|---:|---:|---:|
| E4 | EfficientNet-B0 (ImageNet) | **81.7%** | 95.8% | 0.817 | 6 / 9 | 35 min | 29 ms |
| E3 | MobileNetV2 (ImageNet) | 78.0% | 94.6% | 0.778 | 8 / 11 | 40 min | 22 ms |
| Baseline | Custom CNN | 68.2% | 89.8% | 0.680 | 15 / 15 | 58 min | 40 ms |
| E2 | Custom CNN, batch 16 | 67.8% | 89.3% | 0.676 | 15 / 15 | 62 min | 41 ms |
| E1 | Custom CNN, learning rate 1e-4 | 62.9% | 87.0% | 0.624 | 14 / 15 | 59 min | 47 ms |

The CNNs trained from scratch were still improving at epoch 15; the pretrained models overfit after their best epoch, which early stopping kept. The app uses E4 (`checkpoints/best.pt`).

## Running the experiments

1. Open the notebook in Colab with the badge above and pick a GPU runtime (*Runtime → Change runtime type → T4 GPU*), or import it into a Kaggle notebook (*Settings → Accelerator → GPU*, *Internet on*) and use *Save Version → Save & Run All*, which keeps running with the browser closed.
2. Run the cells from the top. The first cell clones this repository; results and checkpoints are saved to Google Drive (`MyDrive/foodlens`) on Colab and to `/kaggle/working/outputs` on Kaggle.
3. Choose which scenarios to train in `RUN_NOW`. Finished runs are skipped, and an interrupted run continues from its last finished epoch.

## Running the app

On Windows, double-click `run_app.bat`. The first run installs the packages from `requirements.txt`; after that it starts the app and opens it in the browser at http://localhost:8501. Close the window to stop it.

On any system:

```bash
pip install -r requirements.txt
python -m streamlit run app.py
```

The trained EfficientNet-B0 (E4) is included as `checkpoints/best.pt`. Other checkpoints from the notebook can be copied into `checkpoints/` to compare models in the app's *Details* section.

## References

- Bossard, L., Guillaumin, M., & Van Gool, L. (2014). Food-101 – Mining discriminative components with random forests. In *Computer Vision – ECCV 2014* (pp. 446–461). Springer. https://doi.org/10.1007/978-3-319-10599-4_29
- Sandler, M., Howard, A., Zhu, M., Zhmoginov, A., & Chen, L.-C. (2018). MobileNetV2: Inverted residuals and linear bottlenecks. In *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition* (pp. 4510–4520). https://doi.org/10.1109/CVPR.2018.00474
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking model scaling for convolutional neural networks. In *Proceedings of the 36th International Conference on Machine Learning* (pp. 6105–6114). PMLR. https://proceedings.mlr.press/v97/tan19a.html
- Nielsen, J. (1994). *Usability engineering*. Morgan Kaufmann.
- Streamlit. (n.d.). *Streamlit documentation*. https://docs.streamlit.io/
- U.S. Department of Agriculture, Agricultural Research Service. (2024). *USDA Food and Nutrient Database for Dietary Studies 2021–2023*. Food Surveys Research Group. https://fdc.nal.usda.gov/
