# -*- coding: utf-8 -*-
import random

def get_random_colors(number_of_colors):
    return [
    "#"+''.join([random.Random(7*i + j).choice('0123456789ABCDEF') for j in range(6)])
        for i in range(number_of_colors)
    ]

def get_random_colors_dict(keys, rgb=False):
    ans = {}
    for i, key in enumerate(keys):
        color = ''.join([random.Random(7*i + j).choice('0123456789ABCDEF') for j in range(6)])
        if rgb:
            ans[key] = tuple(int(color[i:i+2], 16)/255 for i in (0, 2, 4))
        else:
            ans[key] = "#"+color
    return ans

