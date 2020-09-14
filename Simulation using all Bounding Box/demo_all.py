# --------------------------------------------------------
# SiamMask
# Licensed under The MIT License
# Written by Qiang Wang (wangqiang2015 at ia.ac.cn)
# --------------------------------------------------------
import glob
from tools.test_semua import *

parser = argparse.ArgumentParser(description='PyTorch Tracking Demo')

parser.add_argument('--resume', default='', type=str, required=True,
                    metavar='PATH',help='path to latest checkpoint (default: none)')
parser.add_argument('--config', dest='config', default='config_davis.json',
                    help='hyper-parameter of SiamMask in json format')
# parser.add_argument('--base_path', default='../../data/Video1', help='datasets')
parser.add_argument('--base_path', default='../../data/tennis', help='datasets')
# parser.add_argument('--base_path', default='../../data/coba', help='datasets')
parser.add_argument('--cpu', action='store_true', help='cpu mode')
args = parser.parse_args()

if __name__ == '__main__':
    # Setup device
    device = torch.device('cuda')
    torch.backends.cudnn.benchmark = True

    # Setup Model
    cfg = load_config(args)
    from custom import Custom
    siammask = Custom(anchors=cfg['anchors'])
    if args.resume:
        assert isfile(args.resume), 'Please download {} first.'.format(args.resume)
        siammask = load_pretrain(siammask, args.resume)

    siammask.eval().to(device)

    # Parse Image file
    img_files = sorted(glob.glob(join(args.base_path, '*.jp*')))
    ims = [cv2.imread(imf) for imf in img_files]

    # Select ROI
    cv2.namedWindow("SiamMask", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("SiamMask", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        init_rect = cv2.selectROI('SiamMask', ims[0], False, False)
        x, y, w, h = init_rect
    except:
        exit()

    toc = 0
    for f, im in enumerate(ims):
        tic = cv2.getTickCount()
        if f == 0:  # init
            target_pos = np.array([x + w / 2, y + h / 2])
            target_sz = np.array([w, h])
            ocx = int(x + w / 2)
            ocy = int(y + h / 2)
            centeroid = np.array([ocx,ocy])
            state = siamese_init(im, target_pos, target_sz, siammask, cfg['hp'], device=device)  # init tracker
        elif f > 0:  # tracking
            #siammask tracker
            state1 = siamese_track1(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            # state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track

            location1 = state1['ploygon'].flatten()
            mask1 = state1['mask'] > state1['p'].seg_thr

            im[:, :, 2] = (mask1 > 0) * 255 + (mask1 == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location1).reshape((-1, 1, 2))], True, (255, 0, 0), 3)
            # cv2.imshow('SiamMask', im)

            #Pararellogram
            state2,centeroid2 = siamese_track2(state, im, centeroid, mask_enable=True, refine_enable=True, device=device)  # track
            # state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track

            location2 = state2['ploygon'].flatten()
            mask2 = state2['mask'] > state2['p'].seg_thr

            im[:, :, 2] = (mask2 > 0) * 255 + (mask2 == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location2).reshape((-1, 1, 2))], True, (0, 255, 0), 3)
            # cv2.imshow('SiamMask', im)

            #Proposed
            state3 = siamese_track3(state, im, mask_enable=True, refine_enable=True, device=device)  # track
            # state = siamese_track(state, im, mask_enable=True, refine_enable=True, device=device)  # track

            location3 = state3['ploygon'].flatten()
            mask3 = state3['mask'] > state3['p'].seg_thr

            im[:, :, 2] = (mask3 > 0) * 255 + (mask3 == 0) * im[:, :, 2]
            cv2.polylines(im, [np.int0(location3).reshape((-1, 1, 2))], True, (0, 0, 255), 3)
            
            
            
            
            cv2.imshow('SiamMask', im)




            key = cv2.waitKey(1)
            if key > 0:
                break

        toc += cv2.getTickCount() - tic
    toc /= cv2.getTickFrequency()
    fps = f / toc
    print('SiamMask Time: {:02.1f}s Speed: {:3.1f}fps (with visulization!)'.format(toc, fps))
