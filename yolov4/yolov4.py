# Ismaeel - 1/10


# importing cv2, pil
from math import sqrt
import os
import cv2 as cv
from PIL import Image
import numpy as np
from ast import literal_eval as make_tuple


def classify_rgb(rgb_tuple):

    colors = {(255, 0, 0): (255, 0, 0), # red kit
              (0, 255, 0) : (0, 255,0), # green kit
              (0, 0, 255): (0, 0, 255), # blue kit
              (255,255,255): (255,255,255), # white kit
              (0,0,0): (0,0,0) 
              }
            
    # funny maths to calculate the closest similarity to each category of colour
    manhattan = lambda x,y : abs(x[0] - y[0]) + abs(x[1] - y[1]) + abs(x[2] - y[2]) 
    distances = {k: manhattan(v, rgb_tuple) for k, v in colors.items()}
    color = min(distances, key=distances.get)
    return color


def player_detection(vid_name, filename):

    #checks if we have not already done detection
    if not(os.path.exists(f'frames/{vid_name}/yolo_done.txt')):

        # settup up threshholds, (must be above 40 percent sure)
        Conf_threshold = 0.3
        NMS_threshold = 0.3

        class_name = []

        # opening our co dataset class names
        with open('yolov4/coco.txt', 'r') as f:
            class_name = [cname.strip() for cname in f.readlines()]

        # configuring our yolo net
        net = cv.dnn.readNet('yolov4/yolov4.weights', 'yolov4/yolov4.cfg')

        # setting up our model
        model = cv.dnn_DetectionModel(net)
        model.setInputParams(size=(416, 416), scale=1/255, swapRB=True)

        # getting the frame
        frame = cv.imread(f'frames/{vid_name}/{filename}')

        # getting our class, boxes and accuracy from the model from the image
        classes, scores, boxes = model.detect(frame, Conf_threshold, NMS_threshold)

        # creating blank image for detection
        blank_image = np.zeros((frame.shape[0],frame.shape[1],3), np.uint8)

        '''# creating a ball detection counter
        ball_count = []

        # going through each object and checking if a ball is detected
        for x, object in enumerate(classes):
            if object == 'sports ball':
                #adding balls to list, with its position in detection
                ball_count.append((x, boxes[x][0], boxes[x][1]))
        
        # checking if ball is detected more than once
        if len(ball_count) > 1 and os.path.exists('frames/ball_count.txt'):
            
            # getting the prev ball coordinates
            f = open(f'frames/{vid_name}/ball_count.txt', 'r')
            (prev_x, prev_y) = make_tuple(f.readline())
            
            min = 0
            min_index = 0

            for ball in ball_count:
                # getting distance from each ball to the prev
                dist = sqrt((prev_x - ball[1])^2 + (prev_y - ball[2])^2)

                # updating the min value if its smaller and storing the index
                if dist < min:
                    min = dist
                    min_index = ball[0]

            f = open(f'frames/{vid_name}/ball_count.txt', 'w')
            f.write('('+str(ball_count[min_index][1])+','+ball_count[min_index][2]+')')

            for x, obj in enumerate(classes):
                if obj == 'sports ball':
                    if x != min_index:
                        del classes[x]
                        del boxes[x]
                        del scores[x]
        
        elif len(ball_count) > 1 and not(os.path.exists('frames/ball_count.txt')):

            f = open(f'frames/{vid_name}/ball_count', 'w')
            f.write('('+str(ball_count[0][1])+','+ball_count[0][2]+')')
'''

        #setting a varibale for background image
        background = frame

        # for each label
        for (classid, score, box) in zip(classes, scores, boxes):

            print('DETECTING: ', class_name[classid], 'ACC: ', score)

            #getting height and width of bounding boxes
            width, height = int(box[2]), int(box[3])

            #checking if the detected item is a player or a ball and its small enough to be a player
            if (classid == 0 or classid == 32) and width < 100 and height < 200:

                # defining the label
                label = "%s : %f: " % (class_name[classid], score)

                #calculating the middle of player box
                midpoint = [int((box[0]+(box[0]+box[2]))/2), int((box[1]+(box[1]+box[3]))/2)]

                # drawing midpoint
                #cv.circle(frame, midpoint, 3, colour, 2)

                # getting the rgb value of the players midpoint
                im = Image.open(f'frames/{vid_name}/{filename}') # Can be many different formats.
                pix = im.load()
                value = pix[midpoint[0],midpoint[1]] # Set the RGBA Value of the image (tuple)
                im.close()

                # setting the t shirt colour for the player
                t_colour = classify_rgb(value)

                if classid == 32:
                    # setting a colour
                    t_colour = (192,192,192)

                    #drawing the rectangle
                    cv.circle(background, midpoint, 10, t_colour, -1)

                    #putting in our label
                    cv.putText(background, label, (box[0], box[1]-10),
                                cv.FONT_HERSHEY_COMPLEX, 0.3, t_colour, 1)

                elif classid == 0 and score >= 0.4:
                    #drawing the rectangle
                    cv.rectangle(background, box, t_colour, -1)

                    #putting in our label
                    cv.putText(background, label, (box[0], box[1]-10),
                                cv.FONT_HERSHEY_COMPLEX, 0.3, t_colour, 1)                 


        # overwriting our image and saving a txt file to show detection is complete (for later)
        cv.imwrite(f'frames/{vid_name}/{filename}', background)

    else:
        print('DETECTION ALREADY COMPLETE...')
    