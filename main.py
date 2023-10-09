import subprocess
import PIL
import numpy as np
import cv2 as cv
import pyautogui
import pyscreeze
from PIL import Image
import time

__PIL_TUPLE_VERSION = tuple(int(x) for x in PIL.__version__.split("."))
pyscreeze.PIL__version__ = __PIL_TUPLE_VERSION


def positionWindow(path):
    try:
        subprocess.run(["osascript", path])
    except Exception as e:
        print(f"Error running AppleScript: {str(e)}")


def getMushroomPoints(imageName):
    screen = cv.imread(imageName)
    # converts our screen image to grayscale
    screen_grayscale = cv.cvtColor(screen, cv.COLOR_BGR2GRAY)
    mushroom = cv.imread('mushroom.png', 0)
    res = cv.matchTemplate(screen_grayscale, mushroom, cv.TM_CCOEFF_NORMED)
    threshold = 0.5
    loc = np.where(res >= threshold)
    x_v = loc[1]
    y_v = loc[0]
    points = zip(x_v, y_v)
    points = [(128 * round(x / 128), 128 * round(y / 128)) for (x, y) in points]
    points = set(points)
    if imageName == "top.png" and showOutlines:
        for pt in points:
            cv.rectangle(screen, pt, (pt[0] + 128, pt[1] + 128), (0, 0, 255), 2)
        cv.imwrite('topOutlined.jpg', screen)
    if imageName == "bottom.png" and showOutlines:
        for pt in points:
            cv.rectangle(screen, pt, (pt[0] + 128, pt[1] + 128), (0, 0, 255), 2)
        cv.imwrite('bottomOutlined.jpg', screen)
    return points






def createMatrix(n, points):
    matrix = np.ones((n, n))
    for (x, y) in points:
        matrix[y // 128][x // 128] = 0
    return matrix


# converts this into a lights out problem
# takes 2 states, iterates through each tile, if the tiles are the same, the difference is 0 (light is off)
# if the tiles are different, the difference is 1 (light is on)
def computeDifference(current, target):
    current = np.asarray(current)
    target = np.asarray(target)
    difference = np.subtract(current, target)
    # we are going to turn this into a vector by flattening it
    return np.ravel(np.abs(difference))


# produces a n^2 x n^2 matrix representing how different tiles respond to being touched
def computeEffectMatrix(n):
    matrix = np.zeros((n * n, n * n))
    for row in range(n):
        for col in range(n):
            for tile in range(n * n):
                # essentially if the space is a neighbor of the current row and column
                if abs((tile // n) - row) <= 1 and abs((tile % n) - col) <= 1:
                    matrix[row * n + col][tile] = 1
    return matrix


def solve(A, b):
    x = np.linalg.lstsq(A, b, rcond=None)[0]
    return list(np.round(x))


# change the name of this later
def getSolution(vector, n):
    s = []
    for i in range(len(vector)):
        if vector[i] == 1:
            s.append((i // n, i % n))
    return s


def clickSolution(solution, n):
    pyautogui.moveTo(20, 100)
    pyautogui.mouseDown(button='left')
    pyautogui.mouseUp(button='left')
    time.sleep(1)
    startX = bottomRegions[n - 4][0]
    startY = bottomRegions[n - 4][1]
    offset = 48
    for (row, col) in solution:
        time.sleep(2)
        x = (startX + col * 128 + offset) / 2
        y = (startY + row * 128 + offset) / 2
        pyautogui.moveTo(x, y)
        pyautogui.mouseDown(button='left')
        pyautogui.mouseUp(button='left')
    return 0


def getMatrixLength():
    s = pyautogui.screenshot()
    # determines if a tile is occupying a space, then returns if it does basically
    if s.getpixel((240,300)) == (238, 225, 141, 255):
        return 4
    if s.getpixel((160,300)) == (224, 192, 121, 255):
        return 5
    return 6



# corresponds to n = 4, n = 5, n = 6
topRegions = [(256, 236, 766, 746), (192, 172, 834, 814), (130, 100, 900, 870)]
bottomRegions = [(256, 1006, 766, 1516), (192, 942, 834, 1584), (130, 870, 900, 1640)]

auto = True
showOutlines = True
level = 1
positionWindow("positionScript.scpt")


if auto:
    while True:
        n = getMatrixLength()
        im = pyautogui.screenshot()
        topRegion = topRegions[n - 4]
        bottomRegion = bottomRegions[n - 4]

        topImage = im.crop(topRegion)
        bottomImage = im.crop(bottomRegion)
        topImage.save("top.png")
        bottomImage.save("bottom.png")

        #####
        targetMatrix = createMatrix(n, getMushroomPoints("top.png"))
        currentMatrix = createMatrix(n, getMushroomPoints("bottom.png"))
        A = computeEffectMatrix(n)
        b = computeDifference(targetMatrix, currentMatrix)
        x = solve(A, b)
        solution = getSolution(x, n)
        print("Level",level,"solution:",solution)
        clickSolution(solution, n)
        level += 1
        time.sleep(11)
else:
    n = getMatrixLength()
    im = pyautogui.screenshot()
    topRegion = topRegions[n - 4]
    bottomRegion = bottomRegions[n - 4]

    topImage = im.crop(topRegion)
    bottomImage = im.crop(bottomRegion)
    topImage.save("top.png")
    bottomImage.save("bottom.png")

    #####
    targetMatrix = createMatrix(n, getMushroomPoints("top.png"))
    currentMatrix = createMatrix(n, getMushroomPoints("bottom.png"))
    A = computeEffectMatrix(n)
    b = computeDifference(targetMatrix, currentMatrix)
    x = solve(A, b)
    solution = getSolution(x, n)
    print(solution)
    clickSolution(solution, n)
