import argparse
import os
import shutil
import warnings

import torch
import torch.nn.functional as F
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchmetrics import AUROC, AveragePrecision

from constant import RESIZE_SHAPE, NORMALIZE_MEAN, NORMALIZE_STD, ALL_CATEGORY
from data.mvtec_dataset import MVTecDataset
from model.simstseg import SimSTSeg
from model.metrics import AUPRO, IAPS

warnings.filterwarnings("ignore")


def evaluate(args, category, model, visualizer, global_step=0):
    model.eval()
    with torch.no_grad():
        dataset = MVTecDataset(
            is_train=False,
            mvtec_dir=args.mvtec_path + category + "/test/",
            resize_shape=RESIZE_SHAPE,
            normalize_mean=NORMALIZE_MEAN,
            normalize_std=NORMALIZE_STD,
        )
        dataloader = DataLoader(
            dataset, batch_size=args.bs, shuffle=False, num_workers=args.num_workers
        )
        stenc_IAPS = IAPS().cuda()
        de_stdec_IAPS = IAPS().cuda()
        stenc_AUPRO = AUPRO().cuda()
        de_stdec_AUPRO = AUPRO().cuda()
        stenc_AUROC = AUROC().cuda()
        de_stdec_AUROC = AUROC().cuda()
        stenc_AP = AveragePrecision().cuda()
        de_stdec_AP = AveragePrecision().cuda()
        stenc_detect_AUROC = AUROC().cuda()
        de_stdec_detect_AUROC = AUROC().cuda()
        seg_IAPS = IAPS().cuda()
        seg_AUPRO = AUPRO().cuda()
        seg_AUROC = AUROC().cuda()
        seg_AP = AveragePrecision().cuda()
        seg_detect_AUROC = AUROC().cuda()

        for _, sample_batched in enumerate(dataloader):
            img = sample_batched["img"].cuda()
            mask = sample_batched["mask"].to(torch.int64).cuda()

            output_segmentation, output_stenc, output_de_stdec, output_stenc_list, output_de_stdec_list = model(img)

            output_segmentation = F.interpolate(
                output_segmentation,
                size=mask.size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            output_stenc = F.interpolate(
                output_stenc, size=mask.size()[2:], mode="bilinear", align_corners=False
            )
            output_de_stdec = F.interpolate(
                output_de_stdec, size=mask.size()[2:], mode="bilinear", align_corners=False
            )

            mask_sample = torch.max(mask.view(mask.size(0), -1), dim=1)[0]
            output_segmentation_sample, _ = torch.sort(
                output_segmentation.view(output_segmentation.size(0), -1),
                dim=1,
                descending=True,
            )
            output_segmentation_sample = torch.mean(
                output_segmentation_sample[:, : args.T], dim=1
            )
            output_stenc_sample, _ = torch.sort(
                output_stenc.view(output_stenc.size(0), -1), dim=1, descending=True
            )
            output_de_stdec_sample, _ = torch.sort(
                output_de_stdec.view(output_de_stdec.size(0), -1), dim=1, descending=True
            )
            output_stenc_sample = torch.mean(output_stenc_sample[:, : args.T], dim=1)
            output_de_stdec_sample = torch.mean(output_de_stdec_sample[:, : args.T], dim=1)

            stenc_IAPS.update(output_stenc, mask)
            de_stdec_IAPS.update(output_de_stdec, mask)
            stenc_AUPRO.update(output_stenc, mask)
            de_stdec_AUPRO.update(output_de_stdec, mask)
            stenc_AP.update(output_stenc.flatten(), mask.flatten())
            de_stdec_AP.update(output_de_stdec.flatten(), mask.flatten())
            stenc_AUROC.update(output_stenc.flatten(), mask.flatten())
            de_stdec_AUROC.update(output_de_stdec.flatten(), mask.flatten())
            stenc_detect_AUROC.update(output_stenc_sample, mask_sample)
            de_stdec_detect_AUROC.update(output_de_stdec_sample, mask_sample)

            seg_IAPS.update(output_segmentation, mask)
            seg_AUPRO.update(output_segmentation, mask)
            seg_AP.update(output_segmentation.flatten(), mask.flatten())
            seg_AUROC.update(output_segmentation.flatten(), mask.flatten())
            seg_detect_AUROC.update(output_segmentation_sample, mask_sample)

        iap_stenc, iap90_stenc = stenc_IAPS.compute()
        iap_de_stdec, iap90_de_stdec = de_stdec_IAPS.compute()
        aupro_stenc, ap_stenc, auc_stenc, auc_detect_stenc = (
            stenc_AUPRO.compute(),
            stenc_AP.compute(),
            stenc_AUROC.compute(),
            stenc_detect_AUROC.compute(),
        )
        aupro_de_stdec, ap_de_stdec, auc_de_stdec, auc_detect_de_stdec = (
            de_stdec_AUPRO.compute(),
            de_stdec_AP.compute(),
            de_stdec_AUROC.compute(),
            de_stdec_detect_AUROC.compute(),
        )
        iap_seg, iap90_seg = seg_IAPS.compute()
        aupro_seg, ap_seg, auc_seg, auc_detect_seg = (
            seg_AUPRO.compute(),
            seg_AP.compute(),
            seg_AUROC.compute(),
            seg_detect_AUROC.compute(),
        )

        visualizer.add_scalar("ST_Enc_IAP", iap_stenc, global_step)
        visualizer.add_scalar("ST_Enc_IAP90", iap90_stenc, global_step)
        visualizer.add_scalar("ST_Enc_AUPRO", aupro_stenc, global_step)
        visualizer.add_scalar("ST_Enc_AP", ap_stenc, global_step)
        visualizer.add_scalar("ST_Enc_AUC", auc_stenc, global_step)
        visualizer.add_scalar("ST_Enc_detect_AUC", auc_detect_stenc, global_step)

        visualizer.add_scalar("DeST_Dec_IAP", iap_de_stdec, global_step)
        visualizer.add_scalar("DeST_Dec_IAP90", iap90_de_stdec, global_step)
        visualizer.add_scalar("DeST_Dec_AUPRO", aupro_de_stdec, global_step)
        visualizer.add_scalar("DeST_Dec_AP", ap_de_stdec, global_step)
        visualizer.add_scalar("DeST_Dec_AUC", auc_de_stdec, global_step)
        visualizer.add_scalar("DeST_Dec_detect_AUC", auc_detect_de_stdec, global_step)

        visualizer.add_scalar("SimSTSeg_IAP", iap_seg, global_step)
        visualizer.add_scalar("SimSTSeg_IAP90", iap90_seg, global_step)
        visualizer.add_scalar("SimSTSeg_AUPRO", aupro_seg, global_step)
        visualizer.add_scalar("SimSTSeg_AP", ap_seg, global_step)
        visualizer.add_scalar("SimSTSeg_AUC", auc_seg, global_step)
        visualizer.add_scalar("SimSTSeg_detect_AUC", auc_detect_seg, global_step)

        print("Eval at step", global_step)
        print("================================")
        print("student-Teacher (ST)")
        print("pixel_AUC:", round(float(auc_stenc), 4))
        print("pixel_AP:", round(float(ap_stenc), 4))
        print("PRO:", round(float(aupro_stenc), 4))
        print("image_AUC:", round(float(auc_detect_stenc), 4))
        print("IAP:", round(float(iap_stenc), 4))
        print("IAP90:", round(float(iap90_stenc), 4))
        print()
        print("================================")
        print("Denoising Student-Teacher (DeST)")
        print("pixel_AUC:", round(float(auc_de_stdec), 4))
        print("pixel_AP:", round(float(ap_de_stdec), 4))
        print("PRO:", round(float(aupro_de_stdec), 4))
        print("image_AUC:", round(float(auc_detect_de_stdec), 4))
        print("IAP:", round(float(iap_de_stdec), 4))
        print("IAP90:", round(float(iap90_de_stdec), 4))
        print()
        print("Segmentation Guided Simsiam Student-Teacher (SimSTSeg)")
        print("pixel_AUC:", round(float(auc_seg), 4))
        print("pixel_AP:", round(float(ap_seg), 4))
        print("PRO:", round(float(aupro_seg), 4))
        print("image_AUC:", round(float(auc_detect_seg), 4))
        print("IAP:", round(float(iap_seg), 4))
        print("IAP90:", round(float(iap90_seg), 4))
        print()

        stenc_IAPS.reset()
        de_stdec_IAPS.reset()
        stenc_AUPRO.reset()
        de_stdec_AUPRO.reset()
        stenc_AUROC.reset()
        de_stdec_AUROC.reset()
        stenc_AP.reset()
        de_stdec_AP.reset()
        stenc_detect_AUROC.reset()
        de_stdec_detect_AUROC.reset()
        seg_IAPS.reset()
        seg_AUPRO.reset()
        seg_AUROC.reset()
        seg_AP.reset()
        seg_detect_AUROC.reset()

        return auc_seg, ap_seg


def test(args, category):
    if not os.path.exists(args.log_path):
        os.makedirs(args.log_path)

    run_name = f"SimSTSeg_MVTec_test_{category}"
    if os.path.exists(os.path.join(args.log_path, run_name + "/")):
        shutil.rmtree(os.path.join(args.log_path, run_name + "/"))

    visualizer = SummaryWriter(log_dir=os.path.join(args.log_path, run_name + "/"))

    model = SimSTSeg(ed=True).cuda()

    assert os.path.exists(
        os.path.join(args.checkpoint_path, args.base_model_name + category + ".pckl")
    )
    model.load_state_dict(
        torch.load(
            os.path.join(
                args.checkpoint_path, args.base_model_name + category + ".pckl"
            )
        )
    )

    _, _ = evaluate(args, category, model, visualizer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--num_workers", type=int, default=16)

    parser.add_argument("--mvtec_path", type=str, default="./datasets/mvtec/")
    parser.add_argument("--dtd_path", type=str, default="./datasets/dtd/images/")
    parser.add_argument("--checkpoint_path", type=str, default="./saved_model/")
    parser.add_argument("--base_model_name", type=str, default="SimSTSeg_MVTec_5000_")
    parser.add_argument("--log_path", type=str, default="./logs/")

    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--T", type=int, default=100)  # for image-level inference

    parser.add_argument("--category", nargs="*", type=str, default=ALL_CATEGORY)
    args = parser.parse_args()

    obj_list = args.category
    for obj in obj_list:
        assert obj in ALL_CATEGORY

    with torch.cuda.device(args.gpu_id):
        for obj in obj_list:
            print(obj)
            test(args, obj)
