import timm
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.model_utils import ASPP, BasicBlock, l2_normalize, make_layer


class TeacherNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=True,
            features_only=True,
            out_indices=[1, 2, 3, 4],
        )
        # freeze teacher model
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        self.eval()
        x1, x2, x3, x4 = self.encoder(x)
        return (x1, x2, x3, x4)


class StudentNet_Enc(nn.Module):
    def __init__(self):
        super(StudentNet_Enc, self).__init__()
        self.encoder = timm.create_model(
            "resnet18",
            pretrained=False,
            features_only=True,
            out_indices=[1, 2, 3],
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1, x2, x3 = self.encoder(x)
        return (x1, x2, x3)


class StudentNet_Dec(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.ed = ed
        if self.ed:
            self.decoder_layer4 = make_layer(BasicBlock, 512, 512, 2)
            self.decoder_layer3 = make_layer(BasicBlock, 512, 256, 2)
            self.decoder_layer2 = make_layer(BasicBlock, 256, 128, 2)
            self.decoder_layer1 = make_layer(BasicBlock, 128, 64, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x_list):
        (x1, x2, x3, x4) = x_list
        x = x4
        b4 = self.decoder_layer4(x)
        b3 = F.interpolate(b4, size=x3.size()[2:], mode="bilinear", align_corners=False)
        b3 = self.decoder_layer3(b3)
        b2 = F.interpolate(b3, size=x2.size()[2:], mode="bilinear", align_corners=False)
        b2 = self.decoder_layer2(b2)
        b1 = F.interpolate(b2, size=x1.size()[2:], mode="bilinear", align_corners=False)
        b1 = self.decoder_layer1(b1)
        return (b1, b2, b3)


class SegmentationNet(nn.Module):
    def __init__(self, inplanes=896):
        super().__init__()
        self.res1 = make_layer(BasicBlock, inplanes, 512, 2)
        self.res = make_layer(BasicBlock, 512, 256, 2)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        self.head = nn.Sequential(
            ASPP(256, 256, [6, 12, 18]),
            nn.Conv2d(256, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 1, 1),
        )

    def forward(self, x):
        x = self.res1(x)
        x = self.res(x)
        x = self.head(x)
        x = torch.sigmoid(x)
        return x


class SimSTSeg(nn.Module):
    def __init__(self, ed=True):
        super().__init__()
        self.teacher_net = TeacherNet()
        self.student_enc = StudentNet_Enc()
        self.student_dec = StudentNet_Dec(ed=True)
        self.segmentation_net = SegmentationNet(inplanes=896)

    def forward(self, img_aug, img_origin=None):
        self.teacher_net.eval()

        if img_origin is None:  # for inference
            img_origin = img_aug.clone()

        x1, x2, x3, x4 = self.teacher_net(img_aug)
        outputs_teacher_aug = [l2_normalize(x1), l2_normalize(x2), l2_normalize(x3)]

        outputs_student_enc_aug = [
            l2_normalize(output_s_enc) for output_s_enc in self.student_enc(img_aug)
        ]

        outputs_student_dec_aug = [
            l2_normalize(output_s_dec) for output_s_dec in self.student_dec((x1, x2, x3, x4))
        ]
        output = torch.cat(
            [
                F.interpolate(
                    torch.cat([-output_t * output_s_enc, -output_t * output_s_dec], dim=1),
                    size=outputs_student_dec_aug[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_t, output_s_enc, output_s_dec in zip(outputs_teacher_aug, outputs_student_enc_aug, outputs_student_dec_aug)
            ],
            dim=1,
        )

        output_segmentation = self.segmentation_net(output)

        outputs_student_enc = [
            l2_normalize(output_s) for output_s in self.student_enc(img_origin)
        ]
        outputs_student_dec = outputs_student_dec_aug

        outputs_teacher = [
            l2_normalize(output_t.detach()) for output_t in self.teacher_net(img_origin)
        ]

        output_stenc_list = []
        for output_t, output_s_enc in zip(outputs_teacher, outputs_student_enc):
            a_map = 1 - torch.sum(output_s_enc * output_t, dim=1, keepdim=True)
            output_stenc_list.append(a_map)

        output_de_stdec_list = []
        for output_t, output_s_dec in zip(outputs_teacher, outputs_student_dec):
            a_map = 1 - torch.sum(output_s_dec * output_t, dim=1, keepdim=True)
            output_de_stdec_list.append(a_map)

        output_stenc = torch.cat(
            [
                F.interpolate(
                    output_de_stenc_instance,
                    size=outputs_student_enc[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_stenc_instance in output_stenc_list
            ],
            dim=1,
        )  # [N, 3, H, W]

        output_de_stdec = torch.cat(
            [
                F.interpolate(
                    output_de_stdec_instance,
                    size=outputs_student_dec[0].size()[2:],
                    mode="bilinear",
                    align_corners=False,
                )
                for output_de_stdec_instance in output_de_stdec_list
            ],
            dim=1,
        )  # [N, 3, H, W]

        output_stenc = torch.prod(output_stenc, dim=1, keepdim=True)

        output_de_stdec = torch.prod(output_de_stdec, dim=1, keepdim=True)

        return output_segmentation, output_stenc, output_de_stdec, output_stenc_list, output_de_stdec_list
