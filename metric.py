import argparse
import cv2
import numpy as np
import torch
from piq import psnr, ssim, FID
from piq.feature_extractors import InceptionV3
from src.arcface_torch.backbones import get_model


def compute_generation_quality(fake_image, gt):  # (B*T,3,96,96)   (B*T,3,96,96) cuda
    global global_step
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


def evaluate(model, val_data_loader):
    global global_epoch, global_step
    eval_epochs = 1
    print('Evaluating model for {} epochs'.format(eval_epochs))
    eval_warp_loss,eval_gen_loss = 0.,0.
    count = 0
    psnrs, ssims, fids = [], [], []
    for epoch in range(eval_epochs):
        prog_bar = tqdm(enumerate(val_data_loader), total=len(val_data_loader))
        for step, (T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch,T_mels) in prog_bar:
            #    (B,T,3,H,W)   (B,T,3,H,W)       (B,ref_N,3,H,W)   (B,ref_N,,3,H,W)  (B,T,1,hv,wv)
            model.eval()
            T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch,T_mels = \
                T_frame_img.cuda(non_blocking=True), T_frame_sketch.cuda(non_blocking=True),\
                ref_N_frame_img.cuda(non_blocking=True), ref_N_frame_sketch.cuda(non_blocking=True),T_mels.cuda(non_blocking=True)

            generated_img, wrapped_ref, perceptual_warp_loss, perceptual_gen_loss \
                = model(T_frame_img, T_frame_sketch, ref_N_frame_img, ref_N_frame_sketch,T_mels)  # (B*T,3,H,W)
            perceptual_warp_loss = perceptual_warp_loss.sum()
            perceptual_gen_loss = perceptual_gen_loss.sum()
            # (B*T,3,H,W)
            gt = torch.cat([T_frame_img[i] for i in range(T_frame_img.size(0))], dim=0)  # (B*T,3,H,W)

            eval_warp_loss += perceptual_warp_loss.item()
            eval_gen_loss += perceptual_gen_loss.item()
            count += 1
            #########compute evaluation index ###########
            psnr, ssim, fid = compute_generation_quality(gt, generated_img)
            psnrs.append(psnr)
            ssims.append(ssim)
            fids.append(fid)
        save_sample_images_gen(T_frame_sketch, ref_N_frame_img, wrapped_ref,generated_img, gt, global_step, checkpoint_dir)
        #                         (B,T,3,H,W)  (B,ref_N,3,H,W)  (B*T,3,H,W) (B*T,3,H,W)(B*T,3,H,W)
    psnr, ssim, fid= np.asarray(psnrs).mean(), np.asarray(ssims).mean(), np.asarray(fids).mean()
    print('psnr %.3f ssim %.3f fid %.3f' % (psnr, ssim, fid))
    writer.add_scalar('psnr', psnr, global_step)
    writer.add_scalar('ssim', ssim, global_step)
    writer.add_scalar('fid', fid, global_step)
    writer.add_scalar('eval_warp_loss', eval_warp_loss / count, global_step)
    writer.add_scalar('eval_gen_loss', eval_gen_loss / count, global_step)
    print('eval_warp_loss :', eval_warp_loss / count,'eval_gen_loss', eval_gen_loss / count,'global_step:', global_step)


@torch.no_grad()
def csim(fake_image, gt, weight = './src/arcface_torch/checkpoints/ms1mv3_arcface_r100_fp16.pth', name='r100'): # glint360k_r100.pth
    # 生成、真值
    if fake_image is None:
        raise ValueError('the input generated image does not exist')
        fake_image = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        fake_image = cv2.imread(fake_image)
        fake_image = cv2.resize(fake_image, (112, 112))
    
    if gt is None:
        raise ValueError('the input ground truth image does not exist')
        gt = np.random.randint(0, 255, size=(112, 112, 3), dtype=np.uint8)
    else:
        gt = cv2.imread(gt)
        gt = cv2.resize(gt, (112, 112))

    fake_image = cv2.cvtColor(fake_image, cv2.COLOR_BGR2RGB)
    fake_image = np.transpose(fake_image, (2, 0, 1))
    fake_image = torch.from_numpy(fake_image).unsqueeze(0).float()
    fake_image.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat_fake = net(fake_image).numpy()

    gt = cv2.cvtColor(gt, cv2.COLOR_BGR2RGB)
    gt = np.transpose(gt, (2, 0, 1))
    gt = torch.from_numpy(gt).unsqueeze(0).float()
    gt.div_(255).sub_(0.5).div_(0.5)
    net = get_model(name, fp16=False)
    net.load_state_dict(torch.load(weight))
    net.eval()
    feat_gt = net(gt).numpy()#.T # 转置

    feat_fake = feat_fake[0]
    feat_gt = feat_gt[0]
    # print("feat_fake",feat_fake.shape,type(feat_fake))
    # print("feat_gt",feat_gt.shape,type(feat_gt))

    re_csim = np.dot(feat_fake, feat_gt) / (np.linalg.norm(feat_fake) * np.linalg.norm(feat_gt))
    print("re_csim",re_csim)
    return re_csim


def my_quality(fake_image, gt): 
    if fake_image is None:
        raise ValueError('the input generated image does not exist')
    else:
        fake_image = cv2.imread(fake_image)
        fake_image = torch.from_numpy(fake_image)#/255
        fake_image_tenser = fake_image.unsqueeze(0).permute(0, 3, 1, 2)/255
    
    if gt is None:
        raise ValueError('the input ground truth image does not exist')
    else:
        gt = cv2.imread(gt)
        gt = torch.from_numpy(gt)#/255
        gt_tenser = gt.unsqueeze(0).permute(0, 3, 1, 2)/255

    print("fake_image", fake_image_tenser.shape, type(fake_image_tenser))

    psnr_value = psnr(fake_image_tenser, gt_tenser, reduction='none')
    ssim_value = ssim(fake_image_tenser, gt_tenser, data_range=1., reduction='none')

    fid_metric = FID()
    feature_extractor = InceptionV3() #.cuda()


    # total_images = torch.cat((gt.unsqueeze(0).permute(0, 3, 1, 2), fake_image.unsqueeze(0).permute(0, 3, 1, 2)), 0)
    gt = gt.unsqueeze(0).permute(0, 3, 1, 2)
    fake_image = fake_image.unsqueeze(0).permute(0, 3, 1, 2)
    # print("total_images", total_images.shape)
    print("fake_image", fake_image.shape)

    gt_feat = fid_metric.compute_feats([
            {'images': gt},
        ], feature_extractor=feature_extractor)
    pd_feats = fid_metric.compute_feats([
        {'images': fake_image},
    ], feature_extractor=feature_extractor)

    gt_feat = gt_feat.detach()
    pd_feats = pd_feats.detach()

    gt_feats = gt_feat.cuda()
    pd_feats = pd_feat.cuda()

    fid = fid_metric.compute_metric(pd_feats, gt_feats).item()



    print("psnr_value", psnr_value.shape, type(psnr_value), psnr_value)
    print("psnr_value", ssim_value.shape, type(ssim_value), ssim_value)
    print("psnr_value", fid.shape, type(fid), fid)

    return psnr_value, ssim_value, fid


def evaluate_test(fake_image, gt):
    psnr_value, ssim_value, fid = my_quality(fake_image, gt)
    csim_value = csim(fake_image, gt)
    return psnr_value, ssim_value, fid, csim_value
    
    
