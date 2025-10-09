import papermill as pm

screens = [
    "20250216_AWALS37_Full_screen_n96",
    "20250626_AWALS45_Full_screen_n_96"
]

analyses = ["multiclass"]

for screen in screens:
    for analysis in analyses:
        pm.execute_notebook(
            'train_models.ipynb',
            f'train_models-{screen}-{analysis}.ipynb',
            parameters=dict(SCREEN=screen, ANALYSIS=analysis)
        )