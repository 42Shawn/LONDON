
import torch
import os

import models.WideResNet as WRN
import models.PyramidNet as PYN
import models.ResNet as RN

def load_paper_settings(args):

    WRN_path_new = os.path.join('/home/user/shang/prg_kd/overhaul-distillation-masterWRN28-4V1.pt') #78.37%
    WRN_path_official = os.path.join('/home/user/shang/prg_kd/overhaul-distillation-master/WRN28-4_21.09.pt') #78.91
    WRN_path = os.path.join('/home/user/shang/prg_kd/overhaul-distillation-master/overhaul-distillation-masterWRN28-4V0_WRN28-4_21.pt') #77.61

    Pyramid_path = os.path.join('/home/user/shang/prg_kd/overhaul-distillation-master', 'pyramid200_mixup_15.6.pt')



    if args.paper_setting == 'a':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path_official, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'a2':
        teacher = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        # teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'a2-WRN28-4':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        # state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        # teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'b':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path_new, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'd':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path_official, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = RN.ResNet(depth=56, num_classes=100)

    elif args.paper_setting == 'c':
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        state = torch.load(WRN_path_official, map_location={'cuda:0': 'cpu'})['model']
        teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=2, num_classes=100)

    elif args.paper_setting == 'test':#for training
        teacher = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)
        # state = torch.load(WRN_path, map_location={'cuda:0': 'cpu'})['model']
        # teacher.load_state_dict(state)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'test1':#for training
        teacher = RN.ResNet(depth=56, num_classes=100)
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'test2':#for training
        teacher = WRN.WideResNet(depth=28, widen_factor=10, num_classes=100) #WRN28-10
        student = WRN.WideResNet(depth=16, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'e':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = WRN.WideResNet(depth=28, widen_factor=4, num_classes=100)

    elif args.paper_setting == 'f':
        teacher = PYN.PyramidNet(depth=200, alpha=240, num_classes=100, bottleneck=True)
        state = torch.load(Pyramid_path, map_location={'cuda:0': 'cpu'})['state_dict']
        from collections import OrderedDict
        new_state = OrderedDict()
        for k, v in state.items():
            name = k[7:]  # remove 'module.' of dataparallel
            new_state[name] = v
        teacher.load_state_dict(new_state)
        student = PYN.PyramidNet(depth=110, alpha=84, num_classes=100, bottleneck=False)

    else:
        print('Undefined setting name !!!')

    return teacher, student, args