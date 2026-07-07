import torch
from torch_geometric.nn import fps,knn 
from torch_scatter import scatter_mean 
def build_pyramid(coords, img_feats, targets, in_tissue,meso_ratio=0.25, macro_ratio=0.25):
    device = coords.device
    N = coords.shape[0]

    if N < 16:
        return {
            "micro":{"coords":coords,"feats":img_feats,"targets":targets,"in_tissue":in_tissue},
            "meso":{"coords":coords,"feats":img_feats,"targets":targets,"in_tissue":in_tissue},
            "macro":{"coords":coords,"feats":img_feats,"targets":targets,"in_tissue":in_tissue},
        }
    
    micro_dict = {
        "coords":coords,
        "feats":img_feats,
        "targets":targets,
        "in_tissue":in_tissue, 
    }

    batch_micro = torch.zeros(N,dtype=torch.long,device=device)
    idx_meso = fps(coords,batch_micro,ratio=meso_ratio)
    meso_coords = coords[idx_meso]
    num_meso = meso_coords.shape[0]
    assign_meso = knn(x=meso_coords, y=coords, k=1)
    src_feats = img_feats[assign_meso[0]] 

    meso_feats = scatter_mean(src_feats,assign_meso[1],dim=0,dim_size=num_meso)
    if targets is not None:
        src_targets = targets[assign_meso[0]]
        meso_targets = scatter_mean(src_targets,assign_meso[1],dim=0,dim_size=num_meso)
    else:
        meso_targets = None 
    meso_in_tissue = scatter_mean(in_tissue.float(), assign_meso[1], dim=0, dim_size=num_meso) > 0.0
    

    meso_dict = {
        "coords":meso_coords,
        "feats":meso_feats,
        "targets":meso_targets,
        "in_tissue": meso_in_tissue,
    }

    batch_meso = torch.zeros(num_meso,dtype=torch.long,device=device)
    idx_macro = fps(meso_coords,batch_meso,ratio=macro_ratio)
    macro_coords = meso_coords[idx_macro]
    num_macro = macro_coords.shape[0] 
    assign_macro = knn(x=macro_coords,
                       y=meso_coords,k=1)
    src_meso_feats = meso_dict["feats"][assign_macro[0]]
    macro_feats = scatter_mean(src_meso_feats,assign_macro[1],dim=0, dim_size=num_macro)
    
    if meso_dict["targets"] is not None:
        src_meso_targets = meso_dict["targets"][assign_macro[0]]
        macro_targets = scatter_mean(src_meso_targets,
                                     assign_macro[1],
                                     dim=0,
                                     dim_size=num_macro)
    else:
        macro_targets = None 

    src_meso_in_tissue = meso_dict["in_tissue"][assign_macro[0]]
    macro_in_tissue = scatter_mean(src_meso_in_tissue.float(), assign_macro[1], dim=0, dim_size=num_macro) > 0.0
    macro_dict = {
        "coords":macro_coords,
        "feats":macro_feats,
        "targets":macro_targets,
        "in_tissue": macro_in_tissue,
    }
    
    return {
        "macro":macro_dict,
        "meso":meso_dict,
        "micro":micro_dict,
    }


def sample_micro_subgraphs(micro_dict, points_per_anchor=256, target_ratio=0.15, max_anchors=8):
    coords = micro_dict['coords']
    feats = micro_dict['feats']
    targets = micro_dict['targets']
    in_tissue = micro_dict['in_tissue']
    old_batch = micro_dict['batch']
    device = coords.device

    new_coords, new_feats, new_targets, new_in_tissue,new_batch = [], [], [], [], []
    unique_slides = torch.unique(old_batch)

    for slide_id in unique_slides:
        mask = (old_batch == slide_id)
        c = coords[mask]
        f = feats[mask]
        t = targets[mask]
        in_t = in_tissue[mask]
        N = c.shape[0]

        if N<=points_per_anchor:
            new_coords.append(c)
            new_feats.append(f)
            new_targets.append(t)
            new_in_tissue.append(in_t)
            new_batch.append(torch.full((N,), slide_id, dtype=torch.long, device=device))
            
            continue 

        target_total_points = int(N*target_ratio)
        dynamic_num_anchors = max(1,target_total_points//points_per_anchor)
        dynamic_num_anchors = min(dynamic_num_anchors,max_anchors)

        for _ in range(dynamic_num_anchors):
            anchor_idx = torch.randint(0,N,(1,)).item()
            anchor_coords = c[anchor_idx:anchor_idx+1]
            dist_sq = torch.sum((c-anchor_coords)**2,dim=-1)
            _,topk_idx = torch.topk(dist_sq,k=points_per_anchor,largest=False)

            new_coords.append(c[topk_idx])
            new_feats.append(f[topk_idx])
            new_targets.append(t[topk_idx])
            new_in_tissue.append(in_t[topk_idx])
            new_batch.append(torch.full((points_per_anchor,), slide_id, dtype=torch.long, device=device))

    sampled_micro_dict = {
        'coords': torch.cat(new_coords, dim=0),
        'feats': torch.cat(new_feats, dim=0),
        'targets': torch.cat(new_targets, dim=0),
        'in_tissue':torch.cat(new_in_tissue,dim=0),
        'batch': torch.cat(new_batch, dim=0)
    }

    return sampled_micro_dict


def sample_micro_subgraphs_complexity_aware(micro_dict, points_per_anchor=256, target_ratio=0.15, max_anchors=8, k_neighbors=5):
    coords = micro_dict['coords']
    feats = micro_dict['feats']
    targets = micro_dict['targets']
    old_batch = micro_dict['batch']
    device = coords.device
    new_coords, new_feats, new_targets, new_batch = [], [], [], []
    unique_slides = torch.unique(old_batch)
    for slide_id in unique_slides:
        mask = (old_batch == slide_id)
        c = coords[mask]
        f = feats[mask]
        t = targets[mask] if targets is not None else None
        N = c.shape[0]

        if N <= points_per_anchor:
            new_coords.append(c)
            new_feats.append(f)
            if t is not None: new_targets.append(t)
            new_batch.append(torch.full((N,), slide_id, dtype=torch.long, device=device))
            continue 

        assign = knn(x=c, y=c, k=k_neighbors + 1)
        src_feats = f[assign[0]]
        dst_feats = f[assign[1]]
        feat_diff = torch.norm(src_feats - dst_feats, dim=-1)
        complexity_scores = scatter_mean(feat_diff, assign[0], dim=0, dim_size=N)
        prob_dist = complexity_scores / (complexity_scores.sum() + 1e-8)
        target_total_points = int(N * target_ratio)
        dynamic_num_anchors = max(1, target_total_points // points_per_anchor)
        dynamic_num_anchors = min(dynamic_num_anchors, max_anchors)
        anchor_indices = torch.multinomial(prob_dist, num_samples=dynamic_num_anchors, replacement=False)

        for anchor_idx in anchor_indices:
            anchor_coords = c[anchor_idx:anchor_idx+1]
            dist_sq = torch.sum((c - anchor_coords)**2, dim=-1)
            _, topk_idx = torch.topk(dist_sq, k=points_per_anchor, largest=False)

            new_coords.append(c[topk_idx])
            new_feats.append(f[topk_idx])
            if t is not None: new_targets.append(t[topk_idx])
            new_batch.append(torch.full((points_per_anchor,), slide_id, dtype=torch.long, device=device))

    return {
        'coords': torch.cat(new_coords, dim=0),
        'feats': torch.cat(new_feats, dim=0),
        'targets': torch.cat(new_targets, dim=0) if len(new_targets) > 0 else None,
        'batch': torch.cat(new_batch, dim=0)
    }

    
def sample_micro_stflow_style(micro_dict, min_ratio=0.1, max_ratio=0.6):
    coords = micro_dict['coords']
    feats = micro_dict['feats']
    targets = micro_dict['targets']
    old_batch = micro_dict['batch']
    device = coords.device

    new_coords, new_feats, new_targets, new_batch = [], [], [], []

    for slide_id in old_batch.unique():
        mask = (old_batch == slide_id)
        c = coords[mask]
        f = feats[mask]
        t = targets[mask] if targets is not None else None
        N = c.shape[0]
        if N <= 256:
            new_coords.append(c)
            new_feats.append(f)
            if t is not None:
                new_targets.append(t)
            new_batch.append(torch.full((N,), slide_id, dtype=torch.long, device=device))
            continue
        ratio = torch.empty(1).uniform_(min_ratio, max_ratio).item()
        n_sample = max(256, int(N * ratio))
        n_sample = min(n_sample, N)
        anchor_idx = torch.randint(0, N, (1,)).item()
        dist_sq = ((c - c[anchor_idx:anchor_idx+1]) ** 2).sum(-1)
        _, topk_idx = torch.topk(dist_sq, k=n_sample, largest=False)

        new_coords.append(c[topk_idx])
        new_feats.append(f[topk_idx])
        if t is not None:
            new_targets.append(t[topk_idx])
            
        new_batch.append(torch.full((n_sample,), slide_id, dtype=torch.long, device=device))

    return {
        'coords': torch.cat(new_coords, dim=0),
        'feats': torch.cat(new_feats, dim=0),
        'targets': torch.cat(new_targets, dim=0) if len(new_targets) > 0 else None,
        'batch': torch.cat(new_batch, dim=0),
    }



