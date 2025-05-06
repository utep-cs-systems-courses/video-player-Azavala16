#!/usr/bin/env python3

import threading
import cv2
import time

# configuration
VIDEO_FILE = 'clip.mp4'
cap = cv2.VideoCapture(VIDEO_FILE)
MAX_FRAMES = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
cap.release()
BUFFER_SIZE = 10 # to bound queues at ten frames

buffer1 = [None] * BUFFER_SIZE # buffer for raw frames
buffer2 = [None] * BUFFER_SIZE # buffer for grayscale frames

# buffer indexes
in_index1 = 0
out_index1 = 0
in_index2 = 0
out_index2 = 0

# locks and semaphores for buffer1 (raw frames)
mutex1 = threading.Lock()                   # for exclusive access to buffer of raw frames
empty1 = threading.Semaphore(BUFFER_SIZE)   # to track how many empty slots available in buffer
full1 = threading.Semaphore(0)              # to track currently filled slots in buffer

# locks and semaphores for buffer2 (grayscale frames)
mutex2 = threading.Lock()
empty2 = threading.Semaphore(BUFFER_SIZE)
full2 = threading.Semaphore(0)

# flags to signal when extraction/gray-scaling is complete
extraction_done = False
grayscale_done = False

# producer of raw frames
# will block if buffer1 is full (wait to add more raw frames when consumer releases a slot)
def extract_frames():
    global in_index1, extraction_done
    cap = cv2.VideoCapture(VIDEO_FILE) # open video clip
    count = 0
    success, frame = cap.read() # read a frame
    # loop until all frames are read
    while success and count < MAX_FRAMES:
        print(f'[Extract] Frame {count}')
        empty1.acquire() # fills a slot in the raw frame buffer (reduce available frame slots by 1)

        # only one thread will write frames to buffer1 by using mutex1
        with mutex1:
            buffer1[in_index1] = frame                # add frame into buffer1
            in_index1 = (in_index1 + 1) % BUFFER_SIZE # reset in_index when reaching BUFFER_SIZE for the next 10 frames, mutex prevents race condition
        full1.release()                               # make a frame slot from buffer available for consumer (available frame increases by 1)
        count += 1
        success, frame = cap.read()
    extraction_done = True
    print('[Extract] Done')


# consumer of raw frames, producer of grayscale frames
# will block if buffer1 is empty (wait for raw frames to consume)
# will block if buffer2 is full  (wait to add grayscale frames to display)
def convert_to_grayscale():
    global in_index2, out_index1, grayscale_done
    count = 0
    while True:
        full1.acquire()                                 # wait for a raw frame to become available for consumption, semaphore for available raw frames is reduced by 1
        with mutex1:                                    # use mutex1 again so only one thread accesses the resource at a time
            frame = buffer1[out_index1]                 # the frame is removed from raw frame buffer1
            out_index1 = (out_index1 + 1) % BUFFER_SIZE
        empty1.release()                                # the semaphore for available frame slots is increased by 1

        if frame is None:
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # convert the frame to grayscale
        print(f'[Grayscale] Frame {count}')
        empty2.acquire()                                     # fill a slot in buffer2 for grayscale frames, will wait if buffer full
        with mutex2:
            buffer2[in_index2] = gray_frame                  
            in_index2 = (in_index2 + 1) % BUFFER_SIZE
        full2.release()                                      # the semaphore makes a grayscale frame available for display
        count += 1

        if extraction_done and count >= MAX_FRAMES:
            break
    grayscale_done = True
    print('[Grayscale] Done')


# consumer of grayscale frames
# becomes bloocked if buffer2 is empty
def display_frames():
    global out_index2
    count = 0
    while True:
        full2.acquire()                                      # acquire grayscale frame to display, becomes blocked if buffer2 empty 
        with mutex2:
            frame = buffer2[out_index2]
            out_index2 = (out_index2 + 1) % BUFFER_SIZE
        empty2.release()                                     # enables t2 to add the next grayscale frame into buffer2

        if frame is None:
            break

        print(f'[Display] Frame {count}')
        cv2.imshow('Video', frame)              # display frame from buffer2
        if cv2.waitKey(42) & 0xFF == ord('q'):  # wait 24 fps per frame
            break
        count += 1

        if grayscale_done and count >= MAX_FRAMES:
            break
    print('[Display] Done')
    cv2.destroyAllWindows()

# === Main ===
if __name__ == '__main__':
    t1 = threading.Thread(target=extract_frames)
    t2 = threading.Thread(target=convert_to_grayscale)
    t3 = threading.Thread(target=display_frames)

    t1.start()
    t2.start()
    t3.start()

    t1.join()
    t2.join()
    t3.join()

