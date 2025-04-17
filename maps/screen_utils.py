import numpy as np

def categorize(response):
    if np.issubdtype(response.dtype, np.number):
        return response

    unique_response = np.unique(response)
    ordered = list(sorted(unique_response, key=lambda x: x != "WT"))
    return np.array([ordered.index(r) for r in response])
