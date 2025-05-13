import random

def concat_anno_rc_txt():
    root_dir = 'data_resource/cls_pn/cut_img'
    for mode in ['train','val']:
        with open(f'{root_dir}/anno_{mode}.txt', 'r') as f:
            anno_lines = f.readlines()
        with open(f'{root_dir}/neg_rc_{mode}.txt', 'r') as f:
            neg_lines = f.readlines()
        
        # positive patch in original
        concat_lines = [*anno_lines, *neg_lines]
        random.shuffle(concat_lines)
        with open(f'{root_dir}/{mode}_origin.txt', 'w') as txtf:
            txtf.writelines(concat_lines)
        
        # positive patch in random cut
        with open(f'{root_dir}/rcp_anno_{mode}.txt', 'r') as f:
            rcp_lines = f.readlines()
        anno_neg_lines = []
        for line in anno_lines:
            patch_clsid = line.split(' ')[1].strip()
            if int(patch_clsid) == 0:
                anno_neg_lines.append(line)
        concat_lines = [*anno_neg_lines, *rcp_lines, *neg_lines]
        random.shuffle(concat_lines)
        with open(f'{root_dir}/{mode}_rcp.txt', 'w') as txtf:
            txtf.writelines(concat_lines)

if __name__ == '__main__':
    concat_anno_rc_txt()
