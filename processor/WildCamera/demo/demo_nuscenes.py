import os, torch
from PIL import Image
from WildCamera.newcrfs.newcrf_incidencefield import NEWCRFIF
from tqdm import tqdm

if __name__ == '__main__':
    # NeWCRFs model
    model = NEWCRFIF(version='large07', pretrained=None)
    model.eval()
    model.cuda()

    # Load model checkpoints
    script_dir = os.path.dirname(os.path.realpath(__file__))
    ckpt_path = os.path.join(os.path.dirname(script_dir), 'model_zoo/Release', 'wild_camera_all.pth')
    model.load_state_dict(torch.load(ckpt_path, map_location="cpu"), strict=True)

    images_folder = '/fs/nexus-projects/AD_dashrecon/ADVO/data/nuScenes/NUSCv1.0-mini_12hz/CAM_FRONT/sequences/000/image_2'

    sumfx = 0
    sumfy = 0
    sumcx = 0
    sumcy = 0

    cnt = 0

    for idx, info in enumerate(tqdm(sorted(os.listdir(images_folder)))):
        # imgname, focalgt, source = info.split(' ')
        images_path = os.path.join(images_folder, info)
        intrinsic, _ = model.inference(Image.open(images_path), wtassumption=False)
        focal = intrinsic[0, 0].item()

        fx = intrinsic[0,0].item()
        fy = intrinsic[1,1].item() 
        cx = intrinsic[0,2].item() 
        cy = intrinsic[1,2].item()
        
        sumfx += fx
        sumfy += fy
        sumcx += cx
        sumcy += cy
        cnt += 1
    
    print ([sumfx/cnt, sumfy/cnt, sumcx/cnt, sumcy/cnt])
    # (1298.877039228167, 1359.4259338378906, 825.6013014657157, 488.973758152553)