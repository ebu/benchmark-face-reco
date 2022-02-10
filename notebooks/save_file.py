import numpy as np
import json

def save_data(file_out_np: str, data_np:  np.array,
              file_out_json: str,  metadata: dict) -> None:

    with open(file_out_json + '.json', 'w') as fp:
        json.dump(metadata, fp)

    np.save(file=file_out_np, arr=data_np)
