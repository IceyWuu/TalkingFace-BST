from os.path import join, isfile
from tqdm import tqdm
import torch
from torch import nn
from torch import optim
from tensorboardX import SummaryWriter
from torch.utils import data as data_utils
import numpy as np
from glob import glob
import os, random
import cv2
from piq import psnr, ssim, FID # Icey LPIPS 也在piq
import face_alignment
from piq.feature_extractors import InceptionV3
# from models import define_D
# from loss import GANLoss
from models import Renderer
from models import Landmark_generator as Landmark_transformer
import argparse
parser=argparse.ArgumentParser()
parser.add_argument('--sketch_root',default='/home/zhenglab/wuyubing/TalkingFace-BST/preprocess_result/lrs2_sketch128',\
                    help='root path for sketches') # Icey',required=True'
parser.add_argument('--face_img_root',default='/home/zhenglab/wuyubing/TalkingFace-BST/preprocess_result/lrs2_face128',\
                    help='root path for face frame images') # Icey',required=True'
parser.add_argument('--audio_root',default='/home/zhenglab/wuyubing/TalkingFace-BST/preprocess_result/lrs2_audio',\
                    help='root path for audio mel') # Icey',required=True'
parser.add_argument('--landmark_gen_checkpoint_path', type=str, default='./test/checkpoints/landmarkgenerator_checkpoint.pth')
parser.add_argument('--renderer_checkpoint_path', type=str, default='./test/checkpoints/renderer_checkpoint.pth')
args=parser.parse_args()
# add checkpoints
landmark_gen_checkpoint_path = args.landmark_gen_checkpoint_path
renderer_checkpoint_path =args.renderer_checkpoint_path
#other parameters
num_workers = 20
Project_name = 'batch_inference'   #Project_name
# finetune_path =None
Nl = 15 # Icey 参考图像的landmark，参考图像的数量
ref_N = 25 # 3 # Icey 参考图片，渲染部分
T = 5 # 1 # Icey 推理时每个batch取的帧数
print('Project_name:', Project_name)
# batch_size = 80 # Icey 96       #### batch_size
batch_size_val = 80 # Icey 96    #### batch_size

mel_step_size = 16  # 16
fps = 25
img_size = 128
FID_batch_size = 1024
# evaluate_interval = 1500  #
# checkpoint_interval=evaluate_interval
# lr = 1e-4
# global_step, global_epoch = 0, 0
sketch_root = args.sketch_root
face_img_root = args.face_img_root
filelist_name = 'lrs2'
audio_root=args.audio_root
# checkpoint_root = './checkpoints/renderer/'
# checkpoint_dir = os.path.join(checkpoint_root, 'Pro_' + Project_name)
# reset_optimizer = False
# save_optimizer_state = True
writer = SummaryWriter('tensorboard_runs/Project_{}'.format(Project_name))

#-------------------------------------------------------------------------------
# the following is the index sequence for fical landmarks detected by mediapipe
ori_sequence_idx = [162, 127, 234, 93, 132, 58, 172, 136, 150, 149, 176, 148, 152, 377, 400, 378, 379, 365, 397, 288,
                    361, 323, 454, 356, 389,  #
                    70, 63, 105, 66, 107, 55, 65, 52, 53, 46,  #
                    336, 296, 334, 293, 300, 276, 283, 282, 295, 285,  #
                    168, 6, 197, 195, 5,  #
                    48, 115, 220, 45, 4, 275, 440, 344, 278,  #
                    33, 246, 161, 160, 159, 158, 157, 173, 133, 155, 154, 153, 145, 144, 163, 7,  #
                    362, 398, 384, 385, 386, 387, 388, 466, 263, 249, 390, 373, 374, 380, 381, 382,  #
                    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146,  #
                    78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95]

# the following is the connections of landmarks for drawing sketch image
FACEMESH_LIPS = frozenset([(61, 146), (146, 91), (91, 181), (181, 84), (84, 17),
                           (17, 314), (314, 405), (405, 321), (321, 375),
                           (375, 291), (61, 185), (185, 40), (40, 39), (39, 37),
                           (37, 0), (0, 267),
                           (267, 269), (269, 270), (270, 409), (409, 291),
                           (78, 95), (95, 88), (88, 178), (178, 87), (87, 14),
                           (14, 317), (317, 402), (402, 318), (318, 324),
                           (324, 308), (78, 191), (191, 80), (80, 81), (81, 82),
                           (82, 13), (13, 312), (312, 311), (311, 310),
                           (310, 415), (415, 308)])
FACEMESH_LEFT_EYE = frozenset([(263, 249), (249, 390), (390, 373), (373, 374),
                               (374, 380), (380, 381), (381, 382), (382, 362),
                               (263, 466), (466, 388), (388, 387), (387, 386),
                               (386, 385), (385, 384), (384, 398), (398, 362)])
FACEMESH_LEFT_EYEBROW = frozenset([(276, 283), (283, 282), (282, 295),
                                   (295, 285), (300, 293), (293, 334),
                                   (334, 296), (296, 336)])
FACEMESH_RIGHT_EYE = frozenset([(33, 7), (7, 163), (163, 144), (144, 145),
                                (145, 153), (153, 154), (154, 155), (155, 133),
                                (33, 246), (246, 161), (161, 160), (160, 159),
                                (159, 158), (158, 157), (157, 173), (173, 133)])
FACEMESH_RIGHT_EYEBROW = frozenset([(46, 53), (53, 52), (52, 65), (65, 55),
                                    (70, 63), (63, 105), (105, 66), (66, 107)])
FACEMESH_FACE_OVAL = frozenset([(389, 356), (356, 454),
                                (454, 323), (323, 361), (361, 288), (288, 397),
                                (397, 365), (365, 379), (379, 378), (378, 400),
                                (400, 377), (377, 152), (152, 148), (148, 176),
                                (176, 149), (149, 150), (150, 136), (136, 172),
                                (172, 58), (58, 132), (132, 93), (93, 234),
                                (234, 127), (127, 162)])
FACEMESH_NOSE = frozenset([(168, 6), (6, 197), (197, 195), (195, 5), (5, 4),
                           (4, 45), (45, 220), (220, 115), (115, 48),
                           (4, 275), (275, 440), (440, 344), (344, 278), ])
FACEMESH_CONNECTION = frozenset().union(*[
    FACEMESH_LIPS, FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, FACEMESH_RIGHT_EYE,
    FACEMESH_RIGHT_EYEBROW, FACEMESH_FACE_OVAL, FACEMESH_NOSE
])

full_face_landmark_sequence = [*list(range(0, 4)), *list(range(21, 25)), *list(range(25, 91)),  #upper-half face
                               *list(range(4, 21)),  # jaw
                               *list(range(91, 131))]  # mouth

def summarize_landmark(edge_set):  # summarize all ficial landmarks used to construct edge
    landmarks = set()
    for a, b in edge_set:
        landmarks.add(a)
        landmarks.add(b)
    return landmarks

all_landmarks_idx = summarize_landmark(FACEMESH_CONNECTION)
pose_landmark_idx = \
    summarize_landmark(FACEMESH_NOSE.union(*[FACEMESH_RIGHT_EYEBROW, FACEMESH_RIGHT_EYE,
                                             FACEMESH_LEFT_EYE, FACEMESH_LEFT_EYEBROW, ])).union(
        [162, 127, 234, 93, 389, 356, 454, 323])
# pose landmarks are landmarks of the upper-half face(eyes,nose,cheek) that represents the pose information

content_landmark_idx = all_landmarks_idx - pose_landmark_idx
# content_landmark include landmarks of lip and jaw which are inferred from audio
#-------------------------------------------------------------------------------



# criterionFeat = torch.nn.L1Loss()
class Dataset(object): # Icey 使用DataLoader前，定义自己dataset的写法
    def get_vid_name_list(self, split):
        filelist = []
        with open('filelists/{}/{}.txt'.format(filelist_name, split)) as f:
            for line in f:
                line = line.strip()
                if ' ' in line: line = line.split()[0]
                filelist.append(line)
        return filelist

    def __init__(self, split):
        min_len = 25
        vid_name_lists= self.get_vid_name_list(split)
        self.available_video_names=[]
        print("filter videos with min len of ",min_len,'....')
        for vid_name in tqdm(vid_name_lists,total=len(vid_name_lists)):
            img_paths = list(glob(join(face_img_root,vid_name, '*.png')))
            vid_len=len(img_paths)
            if  vid_len>= min_len:
                self.available_video_names.append((vid_name,vid_len))
        print("complete,with available vids: ", len(self.available_video_names), '\n')

    def normalize_and_transpose(self, window):
        x = np.asarray(window) / 255.
        x = np.transpose(x, (0, 3, 1, 2))
        return torch.FloatTensor(x)  # B,3,H,W
    def __len__(self):
        return len(self.available_video_names)

    def __getitem__(self, idx):
        while 1:
            vid_idx = random.randint(0, len(self.available_video_names) - 1)
            vid_name = self.available_video_names[vid_idx][0]
            vid_len = self.available_video_names[vid_idx][1]
            face_img_paths = list(glob(join(face_img_root,vid_name, '*.png')))

            # 1.randomly select a windows of 5 frame
            window_T=5
            random_start_idx = random.randint(0,vid_len-window_T)
            T_idxs = list(range(random_start_idx, random_start_idx + window_T))

            # 2. read  face image and sketch
            T_face_paths = [os.path.join(face_img_root, vid_name, str(idx) + '.png') for idx in T_idxs]
            ref_N_fpaths = random.sample(face_img_paths, ref_N)


            T_frame_img=[]
            T_frame_sketch = []
            for img_path in T_face_paths:
                sketch_path = os.path.join(sketch_root,
                            '/'.join(img_path.split('/')[-3:]))
                if os.path.isfile(img_path)  and os.path.isfile(sketch_path):
                    T_frame_img.append(cv2.resize(cv2.imread(img_path),(img_size,img_size)))
                    T_frame_sketch.append(cv2.imread(sketch_path))
                else:
                    break
            if len(T_frame_img)!=window_T:  #T (H,W,3)
                continue

            ref_N_frame_img,ref_N_frame_sketch = [],[]
            for img_path in ref_N_fpaths:
                sketch_path = os.path.join(sketch_root,
                                           '/'.join(img_path.split('/')[-3:]))
                if os.path.isfile(img_path)  and os.path.isfile(sketch_path):
                    ref_N_frame_img.append(cv2.resize(cv2.imread(img_path),(img_size,img_size)))
                    ref_N_frame_sketch.append(cv2.imread(sketch_path))
                else:
                    break
            if len(ref_N_frame_img) != ref_N:  # ref_N (H,W,3)
                continue

            T_frame_img = self.normalize_and_transpose(T_frame_img)  #: T,3,H,W
            T_frame_sketch = self.normalize_and_transpose(T_frame_sketch)  #: T,3,H,W

            ref_N_frame_img = self.normalize_and_transpose(ref_N_frame_img)  # ref_N,3,H,W
            ref_N_frame_sketch = self.normalize_and_transpose(ref_N_frame_sketch)  # ref_N,3,H,W

            # 3. get T audio mel
            try:
                audio_mel = np.load(join(audio_root,vid_name, "audio.npy"))
            except Exception as e:
                continue
            frame_idx=T_idxs[2]
            mel_start_frame_idx = frame_idx - 2  ###around the frame idx
            if mel_start_frame_idx < 0:
                continue
            start_idx = int(80. * (mel_start_frame_idx / float(fps)))
            m = audio_mel[start_idx: start_idx + mel_step_size, :]  # get five frame around
            if m.shape[0] != mel_step_size:  # in the end of vid
                continue
            T_mels = m.T  # (hv,wv)
            T_mels = torch.FloatTensor(T_mels).unsqueeze(0).unsqueeze(0)  # (1,1,hv,wv)

            return T_frame_img[2].unsqueeze(0),T_frame_sketch,ref_N_frame_img,ref_N_frame_sketch,T_mels
            #      (1,3,H,W)   (T,3,H,W)       (ref_N,3,H,W)   (ref_N,3,H,W)    (1,1,hv,wv)

def _load(checkpoint_path):
    if device == 'cuda':
        checkpoint = torch.load(checkpoint_path)
    else:
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    return checkpoint

def load_model(model, path):
    print("Load checkpoint from: {}".format(path))
    checkpoint = _load(path)
    s = checkpoint["state_dict"]
    new_s = {}
    for k, v in s.items():
        if k[:6] == 'module':
            new_k=k.replace('module.', '', 1)
        else:
            new_k =k
        new_s[new_k] = v
    model.load_state_dict(new_s)
    model = model.to(device)
    return model.eval()

# def load_checkpoint(path, model, optimizer, reset_optimizer=False, overwrite_global_states=True):
#     global global_step
#     global global_epoch
#     print("Load checkpoint from: {}".format(path))
#     checkpoint = torch.load(path)
#     s = checkpoint["state_dict"]
#     new_s = {}
#     for k, v in s.items():
#         new_s[k.replace('module.', '',1)] = v  #
#     # for k, v in s.items():
#     #     new_s['module.'+k] = v
#     model.load_state_dict(new_s)
#     if not reset_optimizer:
#         optimizer_state = checkpoint["optimizer"]
#         if optimizer_state is not None:
#             print("Load optimizer state from {}".format(path))
#             optimizer.load_state_dict(checkpoint["optimizer"])
#     if overwrite_global_states:
#         global_step = checkpoint["global_step"]
#         global_epoch = checkpoint["global_epoch"]
#     return model
# def save_checkpoint(model, optimizer, step, checkpoint_dir, epoch, prefix=''):
#     checkpoint_path = join(
#         checkpoint_dir, "{}_epoch_{}_checkpoint_step{:09d}.pth".format(prefix, epoch, global_step))
#     if isfile(checkpoint_path):
#         os.remove(checkpoint_path)
#     optimizer_state = optimizer.state_dict() if save_optimizer_state else None
#     torch.save({
#         "state_dict": model.state_dict(),
#         "optimizer": optimizer_state,
#         "global_step": step,
#         "global_epoch": epoch,
#     }, checkpoint_path)
#     print("Saved checkpoint:", checkpoint_path)


# n_layers_D = 3
# num_D = 2
# disc = define_D(input_nc=3, ndf=64, n_layers_D=n_layers_D, norm='instance', use_sigmoid=False, num_D=num_D,
#                     getIntermFeat=True)
# criterionGAN = GANLoss(use_lsgan=True, tensor=torch.cuda.FloatTensor)
# criterion_L1 = nn.L1Loss()
#evaluate index
fid_metric = FID()
feature_extractor = InceptionV3() #.cuda()
def compute_generation_quality(gt, fake_image):  # (B*T,3,96,96)   (B*T,3,96,96) cuda
    # global global_step
    psnr_values = []
    ssim_values = []
    #############PSNR###########
    psnr_value = psnr(fake_image, gt, reduction='none')
    psnr_values.extend([e.item() for e in psnr_value])
    #############SSIM###########
    ssim_value = ssim(fake_image, gt, data_range=1., reduction='none')
    ssim_values.extend([e.item() for e in ssim_value])

    #############FID###########
    B_mul_T = fake_image.size(0)
    total_images = torch.cat((gt, fake_image), 0)
    if len(total_images) > FID_batch_size:
        total_images = torch.split(total_images, FID_batch_size, 0)
    else:
        total_images = [total_images]

    total_feats = []
    for sub_images in total_images:
        sub_images = sub_images.cuda()
        feats = fid_metric.compute_feats([
            {'images': sub_images},
        ], feature_extractor=feature_extractor)
        feats = feats.detach()
        total_feats.append(feats)
    total_feats = torch.cat(total_feats, 0)
    gt_feat, pd_feat = torch.split(total_feats, (B_mul_T, B_mul_T), 0)

    gt_feats = gt_feat.cuda()
    pd_feats = pd_feat.cuda()

    fid = fid_metric.compute_metric(pd_feats, gt_feats).item()
    return np.asarray(psnr_values).mean(), np.asarray(ssim_values).mean(), fid

# def save_sample_images_gen(T_frame_sketch, ref_N_frame_img, wrapped_ref, generated_img, gt, 404, checkpoint_dir): # Icey set global_step=404
#     #                        (B,T,3,H,W)  (B,ref_N,3,H,W)  (B*T,3,H,W) (B*T,3,H,W) (B*T,3,H,W)
#     ref_N_frame_img = ref_N_frame_img.unsqueeze(1).expand(-1, T, -1, -1, -1, -1)  # (B,T,ref_N,3,H,W)
#     ref_N_frame_img = (ref_N_frame_img.cpu().numpy().transpose(0, 1, 2, 4, 5, 3) * 255.).astype(np.uint8)  # ref: (B,T,ref_N,H,W,3)

#     fake_image = torch.stack(torch.split(generated_img, T, dim=0), dim=0)  #(B,T,3,H,W)
#     fake_image = (fake_image.detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)

#     wrapped_ref = torch.stack(torch.split(wrapped_ref, T, dim=0), dim=0)  # (B,T,3,H,W)
#     wrapped_ref = (wrapped_ref.detach().cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)

#     gt = torch.stack(torch.split(gt, T, dim=0), dim=0)  # (B,T,3,H,W)
#     gt = (gt.cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)

#     T_frame_sketch=(T_frame_sketch[:,2].unsqueeze(1).cpu().numpy().transpose(0, 1, 3, 4, 2) * 255.).astype(np.uint8)  # (B,T,H,W,3)

#     folder = join(checkpoint_dir, "samples_step{:09d}".format(404)) # # Icey set global_step=404
#     if not os.path.exists(folder):
#         os.mkdir(folder)
#     collage = np.concatenate((T_frame_sketch, *[ref_N_frame_img[:, :, i] for i in range(ref_N_frame_img.shape[2])], wrapped_ref, fake_image, gt),
#                              axis=-2)
#     for batch_idx, c in enumerate(collage):   # require (B,T,H,W,3)
#         for t in range(len(c)):
#             cv2.imwrite('{}/{}_{}.png'.format(folder, batch_idx, t), c[t])

def evaluate(model, val_data_loader):
    # global global_epoch, global_step
    eval_epochs = 1
    print('Evaluating model for {} epochs'.format(eval_epochs))
    # eval_warp_loss,eval_gen_loss = 0.,0.
    # count = 0
    psnrs, ssims, fids = [], [], []
    for epoch in range(eval_epochs):
        prog_bar = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
        for step, (T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch,T_mels) in prog_bar:
            #    (B,T,3,H,W)   (B,T,3,H,W)       (B,ref_N,3,H,W)   (B,ref_N,,3,H,W)  (B,T,1,hv,wv)
            model.eval()
            T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch,T_mels = \
                T_frame_img.cuda(non_blocking=True), T_frame_sketch.cuda(non_blocking=True),\
                ref_N_frame_img.cuda(non_blocking=True), ref_N_frame_sketch.cuda(non_blocking=True),T_mels.cuda(non_blocking=True)

            generated_img, _, _, _ = model(T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch,T_mels)  # (B*T,3,H,W)
            # perceptual_warp_loss = perceptual_warp_loss.sum()
            # perceptual_gen_loss = perceptual_gen_loss.sum()
            # (B*T,3,H,W)
            gt = torch.cat([T_frame_img[i] for i in range(T_frame_img.size(0))], dim=0)  # (B*T,3,H,W)

            # eval_warp_loss += perceptual_warp_loss.item()
            # eval_gen_loss += perceptual_gen_loss.item()
            # count += 1
            #########compute evaluation index ###########
            psnr, ssim, fid = compute_generation_quality(gt, generated_img)
            psnrs.append(psnr)
            ssims.append(ssim)
            fids.append(fid)
        # save_sample_images_gen(T_frame_sketch, ref_N_frame_img, wrapped_ref,generated_img, gt, 404, checkpoint_dir) # Icey set global_step = 404
        # #                         (B,T,3,H,W)  (B,ref_N,3,H,W)  (B*T,3,H,W) (B*T,3,H,W)(B*T,3,H,W)
    psnr, ssim, fid= np.asarray(psnrs).mean(), np.asarray(ssims).mean(), np.asarray(fids).mean()
    print('psnr %.3f ssim %.3f fid %.3f' % (psnr, ssim, fid))
    writer.add_scalar('psnr', psnr)
    writer.add_scalar('ssim', ssim)
    writer.add_scalar('fid', fid)
    # writer.add_scalar('eval_warp_loss', eval_warp_loss / count, global_step)
    # writer.add_scalar('eval_gen_loss', eval_gen_loss / count, global_step)
    # print('eval_warp_loss :', eval_warp_loss / count,'eval_gen_loss', eval_gen_loss / count,'global_step:', global_step)
if __name__ == '__main__':
    # if not os.path.exists(checkpoint_dir):
    #     os.makedirs(checkpoint_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    # create a model and optimizer
    # model = Renderer().cuda()
    # optimizer = optim.Adam(model.parameters(), lr=lr)

    # load_model 已经把模型移到cuda了
    landmark_generator_model = load_model(
        model=Landmark_transformer(T=T, d_model=512, nlayers=4, nhead=4, dim_feedforward=1024, dropout=0.1),
        path=landmark_gen_checkpoint_path)
    renderer_model = load_model(model=Renderer(), path=renderer_checkpoint_path)

    # if finetune_path is not None:  ###fine tune
    #     load_checkpoint(finetune_path, model, optimizer, reset_optimizer=False, overwrite_global_states=False)

    if torch.cuda.device_count() > 1:
        Landmark_transformer = nn.DataParallel(Landmark_transformer)
        renderer_model = nn.DataParallel(renderer_model)
    #     disc = nn.DataParallel(disc)

    # disc = disc.cuda()
    # disc_optimizer = torch.optim.Adam([p for p in disc.parameters() if p.requires_grad],lr=1e-4, betas=(0.5, 0.999))
    # create dataset
    # train_dataset = Dataset('train')
    val_dataset = Dataset('test')
    # train_data_loader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=batch_size,
    #     shuffle=True,
    #     drop_last=True,
    #     num_workers=num_workers,
    #     pin_memory=True
    # )
    val_data_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size_val,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=True
    )

    with torch.no_grad():
        evaluate(renderer_model, val_data_loader)

print("end")

    # while global_epoch < 9999999999:
        # prog_bar = tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        # running_warp_loss,running_gen_loss= 0.,0.
        # for step, (T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels) in prog_bar:
            #    (B,T,3,H,W)   (B,T,3,H,W)       (B,ref_N,3,H,W)   (B,ref_N,3,H,W) B,T,1,h,w
            # model.train()
            # disc.train()
            # optimizer.zero_grad()
            # disc_optimizer.zero_grad()
            # T_frame_img,T_frame_sketch,ref_N_frame_img,ref_N_frame_sketch,T_mels =\
            #     T_frame_img.cuda(non_blocking=True),T_frame_sketch.cuda(non_blocking=True),\
            #     ref_N_frame_img.cuda(non_blocking=True),ref_N_frame_sketch.cuda(non_blocking=True),T_mels.cuda(non_blocking=True)

            # with torch.no_grad():
            #     # ori_face_img, target_sketches, ref_imgs, ref_img_sketches, T_mels[:, 2].unsqueeze(0))
            #     generated_img, _, _, _ = renderer_model(T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch, T_mels)# (B*T,3,H,W)

            # perceptual_warp_loss=perceptual_warp_loss.sum()
            # perceptual_gen_loss=perceptual_gen_loss.sum()

            # gt = torch.cat([T_frame_img[i] for i in range(T_frame_img.size(0))], dim=0)  # (B*T,3,H,W)
            # # discriminator
            # pred_fake = disc.forward(generated_img.detach()) # Icey 计算了生成图像经过判别器 disc 的前向传播得到的预测结果,detach()确保在计算假设结果的损失时不会影响生成器的梯度更新
            # loss_D_fake = criterionGAN(pred_fake, False)     # Icey 计算假设结果的判别器损失
            # # Real Detection and Loss
            # pred_real = disc.forward(gt.clone().detach())    # Icey 计算真实图像 gt 经过判别器的前向传播得到的预测结果
            # loss_D_real = criterionGAN(pred_real, True)      # Icey 计算真实图像的判别器损失
            # loss_D = (loss_D_fake + loss_D_real).mean() * 0.5 # Icey 总的判别器损失，即生成图像和真实图像的判别器损失的平均值的一半
            #
            # # GAN loss
            # pred_fake = disc.forward(generated_img)
            # loss_G_GAN = criterionGAN(pred_fake, True).mean()
            # # GAN feature matching loss
            # loss_G_GAN_Feat = 0                   # Icey 初始化特征匹配损失为零
            # feat_weights = 4.0 / (n_layers_D + 1) # Icey 计算特征匹配损失的权重
            # D_weights = 1.0 / num_D               # Icey  计算判别器的权重
            # for i in range(num_D):
            #     for j in range(len(pred_fake[i]) - 1):
            #         loss_G_GAN_Feat += D_weights * feat_weights * \
            #                            criterionFeat(pred_fake[i][j], pred_real[i][j].detach()).mean() * 2.5
            # if global_epoch>25:
            #     loss = 2.5*perceptual_warp_loss+4*perceptual_gen_loss+0.1*2.5*loss_G_GAN+ loss_G_GAN_Feat
            # else:
            #     loss = 2.5 * perceptual_warp_loss+0*perceptual_gen_loss
            # loss.backward()  # Icey 计算当前损失对模型参数的梯度
            # optimizer.step() # Icey 根据损失函数的梯度更新模型参数
            # update discriminator weights:
            # loss_D.backward()
            # disc_optimizer.step()
            ##log#
            # running_warp_loss += perceptual_warp_loss.item()
            # running_gen_loss+= perceptual_gen_loss.item()
            # # if global_step % checkpoint_interval == 0:
            # #     save_checkpoint(model, optimizer, global_step, checkpoint_dir, global_epoch, prefix=Project_name)
            # if  global_step % evaluate_interval == 0 or global_step == 100 or global_step == 500:
                # with torch.no_grad():
                #     evaluate(model, val_data_loader)
            # prog_bar.set_description('epoch: %d step: %d running_warp_loss: %.4f running_gen_loss: %.4f' \
            #                          % (global_epoch,global_step, running_warp_loss / (step + 1),running_gen_loss / (step + 1)))
            # writer.add_scalar('running_warp_loss', running_warp_loss / (step + 1), global_step)
            # writer.add_scalar('running_gen_loss', running_gen_loss / (step + 1), global_step)
        #     global_step += 1
        # global_epoch += 1
# print("end")