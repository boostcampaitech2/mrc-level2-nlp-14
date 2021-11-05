# !bin/bash

python3 new_run.py configs/curri01/easy.yaml

rm -r outputs/tu_curri_test_e_aug_pororo_ver2/checkpoint-1000/trainer_state.json

python3 new_run.py configs/curri01/normal.yaml

rm -r outputs/tu_curri_test_n_aug_pororo_ver2/checkpoint-600/trainer_state.json

python3 new_run.py configs/curri01/hard.yaml
