import os.path
import mediapipe as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
import os, traceback
from tqdm import tqdm
import glob
import argparse
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--process_num', type=int, default=6) #number of process in ThreadPool to preprocess the dataset
parser.add_argument('--dataset_video_root', type=str, default='../mvlrs_v1/main') # Icey "../mvlrs_v1/main" # required=True
parser.add_argument('--output_ori_frames_root', type=str, default='../preprocess_result/lrs2_ori_frames')


args = parser.parse_args()

input_mp4_root = args.dataset_video_root
output_ori_frames_root = args.output_ori_frames_root

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

def process_video_file(mp4_path):
    with mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, refine_landmarks=True,
                               min_detection_confidence=0.5) as face_mesh:
        video_stream = cv2.VideoCapture(mp4_path)
        fps = round(video_stream.get(cv2.CAP_PROP_FPS))
        if fps != 25:
            print(mp4_path, ' fps is not 25!!!')
            exit()
        frames = []
        while 1:
            still_reading, frame = video_stream.read()
            if not still_reading:
                video_stream.release()
                break
            frames.append(frame)

        for frame_idx,full_frame in enumerate(frames):
            results = face_mesh.process(cv2.cvtColor(full_frame, cv2.COLOR_BGR2RGB))
            if not results.multi_face_landmarks:
                print("Not detect face!----------------------------------------")
                continue  # not detect

            out_dir = os.path.join(output_ori_frames_root, '/'.join(mp4_path[:-4].split('/')[-2:]))
            os.makedirs(out_dir, exist_ok=True)
            cv2.imwrite(os.path.join(out_dir, str(frame_idx) + '.png'), full_frame)

def mp_handler(mp4_path):
    try:
        process_video_file(mp4_path)
    except KeyboardInterrupt:
        exit(0)
    except:
        traceback.print_exc()


def main():
    print('looking up videos.... ')
    mp4_list = glob.glob(input_mp4_root + '/*/*.mp4')  #example: .../lrs2_video/5536038039829982468/00001.mp4
    # ['/home/zhenglab/wuyubing/TalkingFace-BST/mvlrs_v1/main/6332062124509813446/00059.mp4']
    print('total videos :', len(mp4_list))

    process_num = args.process_num
    print('process_num: ', process_num)
    p_frames = ThreadPoolExecutor(process_num)
    futures_frames = [p_frames.submit(mp_handler, mp4_path) for mp4_path in mp4_list]
    _ = [r.result() for r in tqdm(as_completed(futures_frames), total=len(futures_frames))]
    print("complete task!")

if __name__ == '__main__':
    main()
