import sys
import signal
import time
#from select_folder import GetFolderDialog

import cv2

fps = 5
videoname = "teste_video.mp4"
filename = "teste_vidtime.time"
directory = "Data/Teste_camera/"
video_dir = directory + videoname
file_dir = directory + filename

def signal_term_handler(signal, frame):
    print('Release')
    cap.release()
    cv2.destroyAllWindows()
    sys.exit(0)

def show_webcam(mirror=False):
    # Modify webcam number
    cap = cv2.VideoCapture(1)
    ret, frame = cap.read()
    if ret:
        frame_width = frame.shape[1]
        frame_height = frame.shape[0]
    else:
        return

    # Define the codec and create VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'X264')
    out = cv2.VideoWriter(video_dir, fourcc, fps, (frame_width, frame_height))

    frames = 0

    with open(file_dir, 'w') as f:
        f.write("Video teste\n")

    while True:
        ret_val, img = cap.read()
        if ret_val:
            if mirror:
                img = cv2.flip(img, 1)
            cv2.imshow('My webcam', img)
            out.write(img)
            frames += 1
            with open(file_dir, 'a') as f:
                f.write(str(time.time()) + "\n")
        if cv2.waitKey(1) == 13:
            break  # esc to quit

    cap.release()
    out.release()
    print("Release")
    cap = cv2.VideoCapture(video_dir)
    frame_no = 0

    drive_time = [0,0,0]
    drive_cnt = 0

    while True:
        #cap.set(2, frame_no)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ret_val, img = cap.read()
        if ret_val:
            if mirror:
                img = cv2.flip(img, 1)
            cv2.imshow('My webcam', img)
        arg = cv2.waitKey(1)
        if arg == 27:
            break  # esc to quit
        elif arg == 13:
            for i, file_str in enumerate(open(file_dir, 'r')):
                if i == frame_no + 2:
                    data = file_str.split('\n')[0]
                    drive_time[drive_cnt] = float(data)
            drive_cnt = (drive_cnt+1)%3

            print(drive_time)
            print(drive_cnt)
            print(frame_no+2)

            if drive_cnt == 0:
                mov_percentage = (drive_time[1] - drive_time[0])/(drive_time[2] - drive_time[0])
                print("Mov percentage: ", mov_percentage)
        elif arg == 83:
            frame_no = (frame_no+1)%frames
        elif arg == 81:
            frame_no = (frame_no - 1) % frames
    cap.release()
    cv2.destroyAllWindows()

def main():
    show_webcam(mirror=False)

if __name__ == '__main__':
    main()