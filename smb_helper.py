import torch
from PIL import Image
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


def render(levelLines):
    scale = 16
    path = "C:/Users/debos/Desktop/Current Projects/Display_SMB"
    graphics = {
        # empty locations
        "-": Image.open(path+"/empty.png").convert('RGBA'),

        # Flag
        "^": Image.open(path+"/flag_top.png").convert('RGBA'),
        "f": Image.open(path+"/flag_white.png").convert('RGBA'),
        "I": Image.open(path+"/flag_middle.png").convert('RGBA'),

        # starting location
        "M": Image.open(path+"/mario.png").convert('RGBA'),

        # Enemies
        "E": Image.open(path+"/gomba.png").convert('RGBA'),
            
        # solid tiles
        "X": Image.open(path+"/floor.png").convert('RGBA'),
        "#": Image.open(path+"/solid.png").convert('RGBA'),

        # Question Mark Blocks
        "Q": Image.open(path+"/question_coin.png").convert('RGBA'),

        # Brick Blocks
        "S": Image.open(path+"/brick.png").convert('RGBA'),

        # Coin
        "o": Image.open(path+"/coin.png").convert('RGBA'),

        # Pipes
        "<": Image.open(path+"/tubetop_left.png").convert('RGBA'),
        ">": Image.open(path+"/tubetop_right.png").convert('RGBA'),
        "[": Image.open(path+"/tube_left.png").convert('RGBA'),
        "]": Image.open(path+"/tube_right.png").convert('RGBA'),
        "O": Image.open(path+"/tubetop.png").convert('RGBA'),
        "H": Image.open(path+"/tube.png").convert('RGBA'),
    }
      
    levelLines = levelLines.strip().split('\n')
    height = len(levelLines)
    width = len(levelLines[0])
    decodedMap = []
    exit_x = -1
    exit_y = -1
    '''for y in range(height):
        decodedMap.append([])
        for x in range(width):
            char = levelLines[y][x]
            if char == "F":
                exit_x = x
                exit_y = y
                char = "-"
            if char == "p":
                singlePipe = True
                topPipe = True
                if(x < width - 1 and levelLines[y][x+1] == 'p') or (x > 0 and levelLines[y][x-1] == 'p'):
                    singlePipe = False
                if y > 0 and levelLines[y-1][x] == 'p':
                    topPipe = False
                if singlePipe:
                    if topPipe:
                        char = "O"
                    else:
                        char = "H"
                else:
                    if topPipe:
                        char = "<"
                        if x > 0 and levelLines[y][x-1] == 'p':
                            char = ">"
                    else:
                        char = "["
                        if x > 0 and levelLines[y][x-1] == 'p':
                            char = "]"
            decodedMap[y].append(char)'''
    for y in range(height):
        decodedMap.append([])
        for x in range(width):
            char = levelLines[y][x]
            if char == "F":
                exit_x = x
                exit_y = y
                char = "-"
            if char == "p":
                topPipe = True
                if y > 0 and levelLines[y-1][x] == 'p':
                    topPipe = False

                if topPipe:
                    char = "<"
                    if x > 0 and levelLines[y][x-1] == 'p':
                        char = ">"
                else:
                    char = "["
                    if x > 0 and levelLines[y][x-1] == 'p':
                        char = "]"
            decodedMap[y].append(char)
    if exit_x > 1:
        decodedMap[1][exit_x] = "^"
        decodedMap[2][exit_x - 1] = "f"
    for y in range(2,exit_y+1):
        decodedMap[y][exit_x] = "I"
    lvl_image = Image.new("RGBA", (width*scale, height*scale), (109,143,252,255))
    for y in range(height):
        for tx in range(width):
            x = width - tx - 1
            shift_x = 0
            if decodedMap[y][x] == "f":
                shift_x = 8
            lvl_image.paste(graphics[decodedMap[y][x]], (x*scale + shift_x, y*scale, (x+1)*scale + shift_x, (y+1)*scale))
    return lvl_image