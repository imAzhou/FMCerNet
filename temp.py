def constrain_merged_region(mrx1, mry1, mrx2, mry2, roi_min_size=2048, max_xy=None):
    """
    对合并后的区域坐标进行约束处理
    
    参数:
        mrx1, mry1, mrx2, mry2: 合并后的区域坐标
        roi_min_size: 区域最小尺寸 (默认2048)
        max_xy: 图像最大尺寸 (W, H) 元组
    
    返回:
        约束后的坐标 (mrx1, mry1, mrx2, mry2)
    """
    # 计算当前区域宽度和高度
    width,height = mrx2 - mrx1, mry2 - mry1
    
    # 约束1: 确保宽高不小于roi_min_size
    if width < roi_min_size:
        center_x = (mrx1 + mrx2) // 2
        mrx1 = max(0, center_x - roi_min_size // 2)
        mrx2 = mrx1 + roi_min_size
    if height < roi_min_size:
        center_y = (mry1 + mry2) // 2
        mry1 = max(0, center_y - roi_min_size // 2)
        mry2 = mry1 + roi_min_size
    
    # 约束2: 确保不超过图像边界
    if max_xy is not None:
        W, H = max_xy
        # 检查x2是否超过宽度
        if mrx2 > W:
            mrx1 = max(0, W - roi_min_size)  # 保证最小尺寸
            mrx2 = W
        # 检查y2是否超过高度
        if mry2 > H:
            mry1 = max(0, H - roi_min_size)  # 保证最小尺寸
            mry2 = H
        
    
    return mrx1, mry1, mrx2, mry2