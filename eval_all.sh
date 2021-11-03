# !bin/bash

python3 new_run.py configs/curri_test/main_easy.yaml > main_easy.txt

# rm -r outputs/main_easy

python3 new_run.py configs/curri_test/main_normal.yaml > main_normal.txt

# rm -r outputs/main_normal

python3 new_run.py configs/curri_test/main_hard.yaml > main_hard.txt

# rm -r outputs/main_hard

python3 new_run.py configs/curri_test/aug_main_easy.yaml > aug_main_easy.txt

# rm -r outputs/aug_main_easy

python3 new_run.py configs/curri_test/aug_main_normal.yaml > aug_main_normal.txt

# rm -r outputs/aug_main_normal

python3 new_run.py configs/curri_test/aug_main_hard.yaml > aug_main_hard.txt

# rm -r outputs/aug_main_hard

python3 new_run.py configs/curri_test/aug_tapt_easy.yaml > aug_tapt_easy.txt

# rm -r outputs/aug_tapt_easy

python3 new_run.py configs/curri_test/aug_tapt_normal.yaml > aug_tapt_normal.txt

# rm -r outputs/aug_tapt_normal

python3 new_run.py configs/curri_test/aug_tapt_hard.yaml > aug_tapt_hard.txt