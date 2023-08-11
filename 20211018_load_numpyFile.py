import numpy as np
from glob import glob

# path_emb_test = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20211007_WholeBrain\dms\test\ep60'
# path_emb_test = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20211007_WholeBrain\dms\test\test_ep60'

# path_emb_test = r'Z:\Sumin_Jung\00000000_DATA\4_cELVO(Feautre_embedding)\20211007_WholeBrain\eic\test\eic'

path_emb_test = r'Z:\Stroke\SharingFolder\cELVO\whole_brain_features_rename\test\eic'

list_train_emb_npy = sorted(glob(path_emb_test + '/*.npy'))
list_train_emb_npz = sorted(glob(path_emb_test + '/*.npz'))
list_path_emb = sorted(list_train_emb_npy + list_train_emb_npz)

# 모든 emb를 numpy 1개로 묶어주기
# for i in range(len(list_path_emb[93:110])):
cnt_cnt = 0
for cur_path in list_path_emb: # [199:216]: # AJ223

    # 가장 첫번째일 경우, stack 하지말고 바로 로드
    if cnt_cnt == 0:
        try:
            emb_stack = np.load(cur_path)['arr_0']
        except:
            emb_stack = np.load(cur_path)
    else:
        # 현재 emb 로드
        try:
            emb_cur = np.load(cur_path)['arr_0']
        except:
            emb_cur = np.load(cur_path)
        # emb stack
        emb_stack = np.vstack([emb_stack, emb_cur])

    cnt_cnt = cnt_cnt + 1

print(emb_stack)

import pandas as pd
df = pd.DataFrame(emb_stack)
df.to_csv('Feature파일_AJ121.csv', index=False)