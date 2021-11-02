# !bin/bash

python3 new_run_curri.py configs/curri01/easy.yaml

rm -r outputs/tu_curri_test_e_noaug/trainer_state.json

python3 new_run_curri.py configs/curri01/normal.yaml

rm -r outputs/tu_curri_test_n_noaug/trainer_state.json

python3 new_run_curri.py configs/curri01/hard.yaml
