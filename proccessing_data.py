# ismaeel - 29/09 - 11.43am

# importing cv2    
import cv2
#importing os for folders
import os
#importing numpy for image dimensions
import numpy as np
#import object detection
from cv2 import CascadeClassifier as cc
# importing my custom yolo file
import yolov4.yolov4 as yolo

def vid_to_image(vid):

    if not(os.path.exists(f'frames/{vid}')):

        # capturing the video
        vidcap = cv2.VideoCapture(f'videos/{vid}.mp4')

        # reading the video and recieving if this was successful and the image
        success,image = vidcap.read()

        count = 0

        try:
            os.mkdir(f'frames/{vid}')
        except:
            print('Directory: ', vid, 'Already exists')

        # looping for when the image is captured
        while success:

            # writing the image to a jpeg file in the frames folder 
            cv2.imwrite(f"frames/{vid}/frame%d.jpg" % count, image)     

            # reading the next part of the video
            success,image = vidcap.read()

            # testing whether the frame is read
            # print('Read a new frame: ', success)

            count += 1

    else:
        print('VIDEO ALREADY CONVERTED...')

def image_crop(image_path):

    image = cv2.imread(image_path[1:])

    # recording the size of the origonal image
    width = image.shape[0]
    height = image.shape[1]

    # var for dimensions of crop (h = height, w = width)
    h_t_crop = 290 # t = top
    w_l_crop = 100 # l = left
    h_b_crop = 250 # b = bottom
    w_r_crop = 100 # r = right

    # cropping the image to only show the football pitch
    crop_img = image[h_t_crop:-h_b_crop, w_l_crop:-w_r_crop]

    os.remove(image_path[1:])

    cv2.imwrite(image_path[1:], crop_img)


def proccess_all_images():

    print('GETTING VIDEOS')
    #getting vid names
    vid_list = get_vid_names()
    
    print('CONVERTING VIDEOS TO IMAGES...')
    # getting each vid name and using it to call the function to turn it into images
    for vid in vid_list:
        vid_to_image(vid)
        print(vid)

    # getting the number of folders (videos) in the frames folder
    num_vids = len(vid_list)

    # looping for the number of folders in the frames
    for x in range(0,num_vids):

        if not(os.path.exists(f'frames/{vid_list[x]}/yolo_done.txt')):
            
            print(f'PROCCESSING {vid_list[x]} FRAMES...')

            if os.path.exists(f'frames/{vid_list[x]}/yolo_done.txt'):

                # getting the number of images for each video
                num_images = len([name for name in os.listdir(f'./frames/{vid_list[x]}')])
                num_images -= 1 # taking away the count of the txt file

            else:

                # getting the number of images for each video
                num_images = len([name for name in os.listdir(f'./frames/{vid_list[x]}')])

            # looping for each frame
            for frame in range(0, num_images):

                print(f'STARTING FRAME {frame}...')
                # setting the frame path for each frame
                frame_path = f'/frames/{vid_list[x]}/frame{frame}.jpg'

                print(f'CROPPING FRAME...')
                #cropping the image
                image_crop(frame_path)

                print('DETECTING PLAYERS....')
                yolo.player_detection(vid_list[x], f'frame{frame}.jpg')

                print(f'FRAME{frame} COMPLETE...')
                print('-------------------------')

            f = open(f'frames/{vid_list[x]}/yolo_done.txt', 'w')
            f.write("true")
            f.close()

        else:
            print(f'VIDEO {x} ALREADY PROCESSED...')
    
        ans = input('play video? (Y/N): ')

        if ans.capitalize() == 'Y':
            play_images(vid_list)


def get_vid_names():

    # getting a list of the video titles in the videos folder, to use later
    vid_list = [name[:-4] for name in os.listdir('./videos')]

    # deletes the file mac creates in folders
    for x,vid in enumerate(vid_list):
        if vid == '.DS_S':
            del vid_list[x]

    return vid_list

def play_images(vid_list, ):

    for x in range(0, len(vid_list)):
        # getting the number of images for each video
        num_images = len([name for name in os.listdir(f'./frames/{vid_list[x]}')]) - 1

        for y in range(0, num_images):
            frame = cv2.imread(f'frames/{vid_list[x]}/frame{y}.jpg')
            cv2.imshow('', frame)
            cv2.waitKey(60)
