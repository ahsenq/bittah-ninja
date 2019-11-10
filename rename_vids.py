# %%
import os
import sys
import pandas as pd
# %%
if __name__ == "__main__":
    folder = sys.argv[1]
    for file in [f for f in os.listdir(folder) if '.mp4' in f]:
        newfile = ''.join(file.split('.mp4')) + '.mp4'
        os.rename(os.path.join(folder, file), os.path.join(folder, newfile))

# %%
# df = pd.read_csv('bittah-ninja/week11_all_assigned.csv')
# df.head()

# # %%
# new_files = []
# for file in df.clip_title:
#     newfile = ''.join(file.split('.mp4')) + '.mp4'
#     new_files.append(newfile)
# new_files[:5]

# # %%
# df['clip_title'] = new_files
# df.head()
# # %%
# df.to_csv('bittah-ninja/week11_all_assigned.csv', index=False)

# %%
