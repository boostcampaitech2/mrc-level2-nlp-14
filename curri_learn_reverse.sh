# !bin/bash

python3 new_run_curri.py configs/curri01/hard.yaml

rm -r outputs/tu_curri_test_aug_h_2/trainer_state.json

python3 new_run_curri.py configs/curri01/normal.yaml

rm -r outputs/tu_curri_test_aug_n_2/trainer_state.json

python3 new_run_curri.py configs/curri01/easy.yaml