import torch
import utils

def get_loss_function(params):
    if params['loss'] == 'an_full':
        return an_full
    elif params['loss'] == 'an_full_regression':
        return an_full_regression
    elif params['loss'] == 'an_slds':
        return an_slds
    elif params['loss'] == 'an_ssdl':
        return an_ssdl
    elif params['loss'] == 'an_full_me':
        return an_full_me
    elif params['loss'] == 'an_slds_me':
        return an_slds_me
    elif params['loss'] == 'an_ssdl_me':
        return an_ssdl_me

def neg_log(x):
    return -torch.log(x + 1e-5)

def bernoulli_entropy(p):
    entropy = p * neg_log(p) + (1-p) * neg_log(1-p)
    return entropy

def an_ssdl(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))
    
    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand[inds[:batch_size], class_id]) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss

def an_slds(batch, model, params, loc_to_feats, neg_type='hard'):

    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]

    loc_emb = model(loc_feat, return_feats=True)
    
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    
    num_classes = loc_pred.shape[1]
    bg_class = torch.randint(low=0, high=num_classes-1, size=(batch_size,), device=params['device'])
    bg_class[bg_class >= class_id[:batch_size]] += 1
    
    # data loss
    loss_pos = neg_log(loc_pred[inds[:batch_size], class_id])
    if neg_type == 'hard':
        loss_bg = neg_log(1.0 - loc_pred[inds[:batch_size], bg_class]) # assume negative
    elif neg_type == 'entropy':
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred[inds[:batch_size], bg_class]) # entropy
    else:
        raise NotImplementedError
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss

def an_full(batch, model, params, loc_to_feats, neg_type='hard'):
    
    inds = torch.arange(params['batch_size'])

    loc_feat, _, class_id = batch
    loc_feat = loc_feat.to(params['device'])
    class_id = class_id.to(params['device'])
    
    assert model.inc_bias == False
    batch_size = loc_feat.shape[0]
    
    # create random background samples and extract features
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    
    # get location embeddings
    loc_cat = torch.cat((loc_feat, rand_feat), 0)  # stack vertically
    loc_emb_cat = model(loc_cat, return_feats=True)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    # get predictions for locations and background locations
    loc_pred = torch.sigmoid(model.class_emb(loc_emb))
    loc_pred_rand = torch.sigmoid(model.class_emb(loc_emb_rand))
    
    # data loss
    if neg_type == 'hard':
        loss_pos = neg_log(1.0 - loc_pred) # assume negative
        loss_bg = neg_log(1.0 - loc_pred_rand) # assume negative
    elif neg_type == 'entropy':
        loss_pos = -1 * bernoulli_entropy(1.0 - loc_pred) # entropy
        loss_bg = -1 * bernoulli_entropy(1.0 - loc_pred_rand) # entropy
    else:
        raise NotImplementedError
    loss_pos[inds[:batch_size], class_id] = params['pos_weight'] * neg_log(loc_pred[inds[:batch_size], class_id])
    
    # total loss
    loss = loss_pos.mean() + loss_bg.mean()
    
    return loss

def an_full_regression(batch, model, params, loc_to_feats, neg_type='soft', margin=1.0):
    """
    회귀 기반 contrastive loss 함수
    Args:
        batch: (loc_feat, _, target_val)
        model: location → embedding → 회귀 예측값
        loc_to_feats: 위치 → feature 추출 함수
        neg_type: 'soft' or 'hard'
        margin: negative 쌍 최소 거리 유도값
    """
    # unpack input
    loc_feat, _, target_val = batch
    loc_feat = loc_feat.to(params['device'])  # shape: (B, D)
    target_val = target_val.to(params['device'])  # shape: (B,), float

    batch_size = loc_feat.shape[0]
    
    # 랜덤 위치 샘플링 (negative 후보용)
    rand_loc = utils.rand_samples(batch_size, params['device'], rand_type='spherical')
    rand_feat = loc_to_feats(rand_loc, normalize=False)
    #rand_val = value_to_preds(rand_loc).to(params['device'])  # 예: rand 위치에 대한 예측 chlorophyll 값

    # 전체 concat 후 임베딩 추출
    loc_cat = torch.cat((loc_feat, rand_feat), dim=0)  # (2B, D)
    loc_emb_cat = model(loc_cat, return_feats=True)    # (2B, E)
    loc_emb = loc_emb_cat[:batch_size, :]
    loc_emb_rand = loc_emb_cat[batch_size:, :]
    
    val_pred = model.val_emb(loc_emb)
    val_pred_rand = model.val_emb(loc_emb_rand)

    # 회귀 기반 contrastive loss 계산
    # 거리: feature 거리 / 유사도: 값 차이에 기반
    diff_pos = torch.abs(val_pred - target_val)
    loss_pos = (diff_pos ** 2).mean() 
    
    diff_neg = torch.abs(val_pred_rand - target_val)
    margin_term = torch.clamp(margin - diff_neg, min=0)
    loss_neg = (margin_term ** 2).mean()

    if neg_type == 'soft':
        # soft similarity: 값 차이가 작을수록 가까워야 함
        loss = loss_pos + loss_neg
    elif neg_type == 'hard':
        loss  = loss_pos
    else:
        raise NotImplementedError

    return loss.mean()

def an_full_me(batch, model, params, loc_to_feats):

    return an_full(batch, model, params, loc_to_feats, neg_type='entropy')

def an_full_regression_me(batch, model, params, loc_to_feats):

    return an_full_regression(batch, model, params, loc_to_feats, neg_type='soft')

def an_ssdl_me(batch, model, params, loc_to_feats):
    
    return an_ssdl(batch, model, params, loc_to_feats, neg_type='entropy')

def an_slds_me(batch, model, params, loc_to_feats):
    
    return an_slds(batch, model, params, loc_to_feats, neg_type='entropy')