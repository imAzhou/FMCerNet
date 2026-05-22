from .MIL.HMIL import HMIL
from .MIL.TransMIL import TransMIL
from .MIL.ABMIL import ABMIL
from .MIL.DSMIL import DSMIL
from .MIL.RRT import RRTMIL

allowed_mil_type = ['HMIL', 'ABMIL', 'TransMIL', 'DSMIL', 'RRTMIL']

def get_mil(config):
    mil_type = config.mil_type
    assert mil_type in allowed_mil_type, f'mil_type allowed in {allowed_mil_type}'
    
    mil_model = None
    if mil_type == 'HMIL':
        mil_model = HMIL()
    
    if mil_type == 'ABMIL':
        mil_model = ABMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            classes=config.classes,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            gate=config.gate
        )
    if mil_type == 'TransMIL':
        mil_model = TransMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_classes=config.num_classes,
            classes=config.classes,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            num_heads=config.num_heads,
            num_attention_layers=config.num_attention_layers
        )
    
    if mil_type == 'DSMIL':
        mil_model = DSMIL(
            in_dim=config.in_dim,
            embed_dim=config.embed_dim,
            num_fc_layers=config.num_fc_layers,
            dropout=config.dropout,
            attn_dim=config.attn_dim,
            dropout_v=config.dropout_v,
            num_classes=config.num_classes,
            classes=config.classes,
        )
    
    if mil_type == 'RRTMIL':
        mil_model = RRTMIL(
            in_dim=config.in_dim,
            mlp_dim=config.mlp_dim,
            embed_dim=config.embed_dim,
            act=config.act,
            num_classes=config.num_classes,
            dropout=config.dropout,
            pos_pos=config.pos_pos,
            pos=config.pos,
            peg_k=config.peg_k,
            attn=config.attn,
            pool=config.pool,
            region_num=config.region_num,
            n_layers=config.n_layers,
            n_heads=config.n_heads,
            multi_scale=config.multi_scale,
            drop_path=config.drop_path,
            da_act=config.da_act,
            trans_dropout=config.trans_dropout,
            ffn=config.ffn,
            ffn_act=config.ffn_act,
            mlp_ratio=config.mlp_ratio,
            da_gated=config.da_gated,
            da_bias=config.da_bias,
            da_dropout=config.da_dropout,
            trans_dim=config.trans_dim,
            n_cycle=config.n_cycle,
            epeg=config.epeg,
            min_region_num=config.min_region_num,
            qkv_bias=config.qkv_bias,
            shift_size=config.shift_size,
            no_norm=config.no_norm,
            classes=config.classes,
        )
    
    return mil_model