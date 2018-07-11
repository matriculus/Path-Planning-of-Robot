import cv2
import numpy as np
import time
from skimage.measure import compare_ssim as ssim
import astarsearch
import traverse_image

print("Path Planning of Robot!")

def main(image_file):

    # Arrays for storing the occupied grid address and planned path
    occupiedGrids = []
    pathPlanned = {}

    # Loading the image
    image = cv2.imread(image_file)
    imW, imH = 60, 60

    obstacles = []
    index = [1, 1]

    # For path, creating a blank matrix of 0's.
    blank_image = np.zeros((imW, imH, 3), dtype=np.uint8)

    # creating an empty array for 100 blank images
    image_list = [[blank_image for _ in range(10)] for _ in range(10)]

    # creating a list of maze
    maze = [[0 for _ in range(10)] for _ in range(10)]

    for x, y, window in traverse_image.sliding_window(image, stepSize=60, windowSize=(imW, imH)):
        # if the window size does not meet the desired window size, it is ignored
        if window.shape[0] != imW or window.shape[1] != imH:
            continue
        
        # cloning the actual image
        clone = image.copy()
        cv2.rectangle(clone, (x, y), (x+imW, y+imH), (0, 255, 0), 2)
        crop_image = image[x:x+imW, y:y+imH]
        image_list[index[0]-1][index[1]-1] = crop_image.copy()

        # printing occupied grids
        average_color = np.uint8(np.average(np.average(crop_image, axis=0), axis=0))

        # iterating through colour matrix
        # getting occupied grids
        if any(i <= 240 for i in average_color):
            maze[index[0]-1][index[1]-1] = 1
            occupiedGrids.append(tuple(index))
        
        # checking for black colour grids
        if any(i <= 20 for i in average_color):
            # print("Obstacles")
            obstacles.append(tuple(index))
        
        # Showing the iteration
        cv2.imshow("Window", clone)
        cv2.waitKey(1)
        time.sleep(0.025)

        # Iterate for the next one
        index[1] += 1
        if index[1]>10:
            index[0] += 1
            index[1] = 1
        
    # getting object list
    # getting occupied grids without black
    list_coloured_grids = [i for i in occupiedGrids if i not in obstacles]

    for start_image in list_coloured_grids:
        key_start_image = start_image
        
        # starting image
        img1 = image_list[start_image[0]-1][start_image[1]-1]

        for grid in [n for n in list_coloured_grids if n != start_image]:
            #next image
            img = image_list[grid[0]-1][grid[1]-1]
            #convert to grayscale
            image = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
            image2 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            #compare structural similarity
            s = ssim(image, image2)
            #if they are similar
            if s > 0.9:
                #perform a star search between both
                result = astarsearch.astar(maze,(start_image[0]-1,start_image[1]-1),(grid[0]-1,grid[1]-1))
                list2=[]
                for t in result:
                    x,y = t[0],t[1]
                    #Contains min path + start_image + endimage
                    list2.append(tuple((x+1,y+1)))
                    #Result contains the minimum path required
                    result = list2

                if not result:	#If no path is found;
                    pathPlanned[start_image] = list(["NO PATH",[], 0])
                
                pathPlanned[start_image] = list([str(grid),result,len(result)])
    
    for obj in list_coloured_grids:
        if  obj not in pathPlanned:	#If no matched object is found;
            pathPlanned[obj] = list(["NO MATCH",[],0])			

    return occupiedGrids, pathPlanned


if __name__ == '__main__':
    image_filename = "images/test_image4.jpg"

    occupied_grids, planned_path = main(image_filename)
    print("Occupied Grids : ")
    print(occupied_grids)
    print("Planned Path :")
    print(planned_path)
    
    cv2.waitKey(0)
    cv2.destroyAllWindows()