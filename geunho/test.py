from data.dataset import ColorHintDataset
import torch
import torch.utils.data as data
import cv2
import tqdm
import os
from data.transform import tensor2im
from model.res_unet.res_unet import ResUnet
from model.res_unet.res_unet_plus import ResUnetPlusPlus
from model.res_unet.unet import UNet

def main():
    # Change to your data root directory
    root_path = "./cv_project"
    check_point = './checkpoints/modelname.pt'
    # Depend on runtime setting
    use_cuda = False

    test_dataset = ColorHintDataset(root_path, 256, "test")

    dataloaders = {}
    dataloaders['test'] = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=True)

    print('test dataset: ', len(test_dataset))

    model = ResUnetPlusPlus(3)

    # model = Unet().cuda()
    model.load_state_dict(torch.load(check_point))
    os.makedirs('outputs/predict', exist_ok=True)

    print('test dataset: ', len(dataloaders['test']))

    model.eval()
    for i, data in enumerate(tqdm.tqdm(dataloaders['test'])):
        if use_cuda:
            l = data["l"].to('cuda')
            ab = data["ab"].to('cuda')
            hint = data["hint"].to('cuda')
            mask = data["mask"].to('cuda')
        else:
            l = data["l"]
            ab = data["ab"]
            hint = data["hint"]
            mask = data["mask"]

        filename = data["file_name"]

        hint_image = torch.cat((l, ab, mask), dim=1)

        output_hint = model(hint_image)
        out_hint_np = tensor2im(output_hint)
        output_bgr = cv2.cvtColor(out_hint_np, cv2.COLOR_LAB2BGR)

        fname = str(filename).replace("['", '')
        fname = fname.replace("']", '')

        cv2.imwrite('./outputs/predict/' + str(fname), output_bgr)

if __name__ == '__main__':
    main()