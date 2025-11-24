import papermill as pm
from itertools import product

screens = ["20250216_AWALS37_Full_screen_n96"]
analyses = ["multiclass_loocv"]
antibodies = [["HSP70/SOD1", "FUS/EEA1", "COX IV/Galectin3/atubulin"]]
markers = ["all"]
grid = list(product(screens, analyses, antibodies, markers))

for params in grid:
    screen, analysis, antibody, marker = params
    ab_string = "_".join(antibody).replace("/", "-")
    pm.execute_notebook(
        'train_models.ipynb',
        f'train_models-{screen}-{analysis}-{ab_string}-{marker}.ipynb',
        parameters=dict(SCREEN=screen, ANALYSIS=analysis, ANTIBODY=antibody, MARKER=marker)
    )