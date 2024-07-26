import torch


def batchify_ray(model, rays, bboxes, feature_maps, Rts = None, global_ts = None, skeletons = None,  vertices = None, scene = None, sample_index_val = None, chuncks = 1024*10, near_far=None, scale=None):
    N = rays.size(0)

    if N <chuncks:
        return model(rays, bboxes, feature_maps,
                    Rts = Rts, global_ts = global_ts, skeletons = skeletons, vertices = vertices, scene = scene, 
                    sample_depth = sample_index_val, near_far =near_far, scale = scale)

    else:
        rays = rays.split(chuncks, dim=0)
        if bboxes is not None:
            bboxes = bboxes.split(chuncks, dim=0)
        else:
            bboxes = [None]*len(rays)
        
        if Rts is not None:
            Rts = Rts.split(chuncks, dim=0)
        else:
            Rts = [None]*len(rays)
        
        if global_ts is not None:
            global_ts = global_ts.split(chuncks, dim=0)
        else:
            global_ts = [None]*len(rays)

        if skeletons is not None:
            skeletons = skeletons.split(chuncks, dim=0)
        else:
            skeletons = [None]*len(rays)

        if near_far is not None:
            near_far = near_far.split(chuncks, dim=0)
        else:
            near_far = [None]*len(rays)

        if sample_index_val is not None:
            sample_index_val = sample_index_val.split(chuncks, dim=0)
        else:
            sample_index_val = [None]*len(rays)

        colors = [[],[]]
        depths = [ [],[]]
        acc_maps = [ [],[]  ]

        ray_masks = []

        for i in range(len(rays)):
            stage2, stage1, ray_mask = model(rays[i], bboxes[i], feature_maps, 
                                            Rts = Rts[i], global_ts = global_ts[i], skeletons = skeletons[i], vertices = vertices,  scene = scene, 
                                            sample_depth = sample_index_val[i], near_far = near_far[i], scale = scale)
            colors[0].append(stage1[0])
            depths[0].append(stage1[1])
            acc_maps[0].append(stage1[2])

            colors[1].append(stage2[0])
            depths[1].append(stage2[1])
            acc_maps[1].append(stage2[2])

            if ray_mask is not None:
                ray_masks.append(ray_mask)

        colors[0] = torch.cat(colors[0], dim=0)
        depths[0] = torch.cat(depths[0], dim=0)
        acc_maps[0] = torch.cat(acc_maps[0], dim=0)

        colors[1] = torch.cat(colors[1], dim=0)
        depths[1] = torch.cat(depths[1], dim=0)
        acc_maps[1] = torch.cat(acc_maps[1], dim=0)

        if len(ray_masks)>0:
            ray_masks = torch.cat(ray_masks, dim=0)

        return (colors[1], depths[1], acc_maps[1], ), (colors[0], depths[0], acc_maps[0]), ray_masks
