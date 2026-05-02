import torch
import numpy as np


def to_text(tns):
    tns = tns.transpose(0,2).transpose(0,1)
    level = []
    for line in tns:
        row = []
        for item in line:
            _, index = torch.max(item,0)
            if index == 0: ele = '-'    # empty/sky  
            elif index == 1: ele = 'X'  # floor/soild
            elif index == 2: ele = 'S'  # brick
            elif index == 3: ele = 'Q'  # collecteble
            elif index == 4: ele = 'E'  # enemy
            elif index == 5: ele = 'o'  # coin
            elif index == 6: ele = 'p'  # pipe  
            else: print("Something went wrong.")
            row.append(ele)
        level.append(row)
    level = np.asarray(level)
    return level

def to_tnsor(tns, channels=7):
    tns = tns.transpose(0,2).transpose(0,1)
    lvl_arr = []
    for line in tns:
        row = []
        for item in line:
            oh = [0] * channels
            _, index = torch.max(item,0)
            oh[index] = 1
            row.append(oh)
        lvl_arr.append(row)
    lvl_tns = torch.tensor(lvl_arr).float()
    return lvl_tns.transpose(0,1).transpose(0,2)

def concat_segments(arr):
    lvl = arr[0]
    for i in range(1,len(arr)):
        lvl = np.concatenate((lvl,arr[i]),axis=1)
    width = len(lvl[0])
    height = len(lvl)
    result = ""
    for y in range(height):
        if y==height-1:
            result += 'XXX'
        elif y==height-2:
            result += '-M-'
        else:
            result += '---'
            
        for x in range(width):
            result += lvl[y][x]
            
        if y==height-1:
            result += 'XXX'
        elif y==height-2:
            result += '-F-'
        else:
            result += '---'
        result += '\n'
    return result


def str_lvl(lvl):
    width = len(lvl[0])
    height = len(lvl)
    result = ""
    for y in range(height):          
        for x in range(width):
            result += lvl[y][x]
        result += '\n'
    return result

def convert_lvl(lvl):
    converted_lvl = []
    for y in range(len(lvl)):
        row = []
        for x in range(len(lvl[y])):
            item = lvl[y][x]
            if item == 'p':
                topPipe = True
                if y > 0 and lvl[y-1][x] == 'p':
                    topPipe = False

                if topPipe:
                    char = "<"
                    if x > 0 and lvl[y][x-1] == 'p':
                        char = ">"
                else:
                    char = "["
                    if x > 0 and lvl[y][x-1] == 'p':
                        char = "]"
            else:
                char = item
            row.append(char)
        converted_lvl.append(row)   
    return converted_lvl
