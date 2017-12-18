import numpy as np
import cv2

# Define the codec and create VideoWriter object (note isColor is False for Gray)
fourcc = cv2.VideoWriter_fourcc(*'DIVX')
out = cv2.VideoWriter('motiondetect.avi',fourcc, 60.0, (1920,1080), isColor = False)

# Look for the first Videosource, define the camerasettings and create HD-VideoCapture object
cap = cv2.VideoCapture(0) 
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT,1080)
cap.set(cv2.CAP_PROP_FPS, 30)

# Define gaussian mixture-based background/foreground segmentation object
foreground = cv2.createBackgroundSubtractorMOG2(detectShadows = False)

while(True):
    # Get camframe in HD
    (ret, camframe) = cap.read()
    # No colors
    grayframe = cv2.cvtColor(camframe, cv2.COLOR_BGR2GRAY)
    # Resize it
    smallframe = cv2.resize(grayframe, (80,60))    
    # Denoising
    blurframe = cv2.medianBlur(smallframe, 3)
    # Get motion
    motionframe = foreground.apply(blurframe)
    # Show some
    cv2.imshow('motionframe',motionframe)
    cv2.imshow('blurframe',blurframe)
    # Threshold
    detect = (np.sum(motionframe))//255
    if detect > 16:
        print("moving object size = ", detect)
        #write HD-stream to .avi -file
        out.write(grayframe)
    k = cv2.waitKey(1) & 0xff
    # For more camerasettings type "s"
    if k == ord('s'):
        cap.set(cv2.CAP_PROP_SETTINGS, 0)
    # Stop it with the SpaceBar or ESC
    if k == ord(' ') or k == 27 or ret == False:
        break

cap.release()
out.release()
cv2.destroyAllWindows()