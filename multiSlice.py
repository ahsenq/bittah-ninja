#!usr/bin/env python3

import os
import subprocess

for item in os.listdir('/root/raw_vids'):
    # print(item)
    sliceWhat = '/root/raw_vids/' + item
    print('sh /root/bittah-ninja/sliceVid.sh ' + sliceWhat)
    folder = item.split('.')[0]
    loot_to_move = '/root/raw_vids/' + folder + '/*_slice* /root/vids'
    print('cp ' + loot_to_move)
    cleanup = 'rm -r ' + '/root/raw_vids/' + folder
    print(cleanup)
    trash = 'rm /root/raw_vids/' + item
    print(trash)


