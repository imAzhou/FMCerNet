import torch
import torch.nn as nn
import numpy as np
import cv2
from cellpose.vit_sam import Transformer
from cellpose import transforms, dynamics, utils, plot
from cellpose.core import run_net
from cerwsi.utils import flow2cellprob,inst2bboxes

class CellposeNet(nn.Module):
    def __init__(self, cell_config, userle=False):
        super(CellposeNet, self).__init__()
        self.dtype = torch.bfloat16
        self.cell_config = cell_config
        self.net = Transformer(dtype=torch.bfloat16)
        self.userle = userle

    @property
    def device(self):
        return next(self.parameters()).device
    
    def load_ckpt(self, ckpt):
        state_dict = torch.load(ckpt, map_location = self.device, weights_only=True)
        keys = [k for k in state_dict.keys()]
        if keys[0][:7] == "module.":
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:] # remove 'module.' of DataParallel/DistributedDataParallel
                new_state_dict[name] = v
            load_result = self.net.load_state_dict(new_state_dict, strict = False)
        else:
            load_result = self.net.load_state_dict(state_dict, strict = False)
        print('Load CellposeSAM: ' + str(load_result))

        if self.dtype != torch.float32:
            self = self.to(self.dtype)
    
    def forward(self, image: np.ndarray, batchsize:int):
        '''
        Args: 
            image: np.ndarray, shape is (H,W,C), RGB image
        '''
        object_list = []
        for ctype,config in self.cell_config.items():
            flowThr,dia = config['flowThr'],float(config['dia'])
            cellprobThr,minSize = config['cellprobThr'], config['min_size']
            results, styles = self.eval(
                image, 
                batch_size=batchsize, 
                diameter=dia)
            flowi, dP, cellprob = results

            if ctype == 'cytoplasm' or ctype == 'nucleus':
                cellprob,boundary_mask = flow2cellprob(dP)
                cellprob[boundary_mask] = 0.
                maski = dynamics.resize_and_compute_masks(
                        dP, cellprob,
                        cellprob_threshold=cellprobThr,
                        flow_threshold=flowThr, resize=None,
                        min_size=minSize, max_size_fraction=0.9,
                        device=self.device)
            else:
                cellprob,boundary_mask = flow2cellprob(dP)
                cellprob[boundary_mask] = 0.
                binary = (cellprob > cellprobThr).astype(np.uint8)
                num_labels, labels = cv2.connectedComponents(binary, connectivity=8)
                maski = labels.astype(np.int32)
                maski = utils.fill_holes_and_remove_small_masks(maski, min_size=minSize)

            object_list.extend(inst2bboxes(maski, userle=self.userle))
        
        return object_list
    
    def eval(self, x, batch_size=8, diameter=None,
             tile_overlap=0.1, bsize=256, augment=True):
        
        x = x[np.newaxis, ...]
        B,Ly_0,Lx_0,C = x.shape
        image_scaling = None
        if diameter is not None:
            image_scaling = 30. / diameter
            x = transforms.resize_image(x,
                Ly=int(x.shape[1] * image_scaling),
                Lx=int(x.shape[2] * image_scaling))
        
        normalize_params = {
            "lowhigh": None,
            "percentile": None,
            "normalize": True,
            "norm3D": True,
            "sharpen_radius": 0,
            "smooth_radius": 0,
            "tile_norm_blocksize": 0,
            "tile_norm_smooth3D": 1,
            "invert": False
        }
        x = transforms.normalize_img(x, **normalize_params)
        dP, cellprob, styles = self._run_net(
            x, 
            augment=augment, 
            batch_size=batch_size, 
            tile_overlap=tile_overlap, 
            bsize=bsize)
        # upsample flows before computing them: 
        dP = self._resize_gradients(dP, to_y_size=Ly_0, to_x_size=Lx_0, to_z_size=None)
        cellprob = self._resize_cellprob(cellprob, to_x_size=Lx_0, to_y_size=Ly_0, to_z_size=None)
        dP, cellprob = dP.squeeze(), cellprob.squeeze()

        # undo resizing:
        dP = self._resize_gradients(dP, to_y_size=Ly_0, to_x_size=Lx_0, to_z_size=None) # works for 2 or 3D: 
        cellprob = self._resize_cellprob(cellprob, to_x_size=Lx_0, to_y_size=Ly_0, to_z_size=None)

        return [plot.dx_to_circ(dP), dP, cellprob], styles
    
    def _run_net(self, x, 
                augment=False, 
                batch_size=8, tile_overlap=0.1,
                bsize=224):
        """ run network on image x """
        yf, styles = run_net(self.net, x, bsize=bsize, augment=augment,
                            batch_size=batch_size,  
                            tile_overlap=tile_overlap, 
                            )
        cellprob = yf[..., -1]
        dP = yf[..., -3:-1].transpose((3, 0, 1, 2))
        if yf.shape[-1] > 3:
            styles = yf[..., :-3]
        styles = styles.squeeze()
        return dP, cellprob, styles
    
    def _resize_cellprob(self, prob: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        """
        Resize cellprob array to specified dimensions for either 2D or 3D.

        Parameters:
            prob (numpy.ndarray): The cellprobs to resize, either in 2D or 3D. Returns the same ndim as provided.
            to_y_size (int): The target size along the Y-axis.
            to_x_size (int): The target size along the X-axis.
            to_z_size (int, optional): The target size along the Z-axis. Required
                for 3D cellprobs.

        Returns:
            numpy.ndarray: The resized cellprobs array with the same number of dimensions
            as the input.

        Raises:
            ValueError: If the input cellprobs array does not have 3 or 4 dimensions.
        """
        prob_shape = prob.shape
        prob = prob.squeeze()
        squeeze_happened = prob.shape != prob_shape
        prob_shape = np.array(prob_shape)

        if prob.ndim == 2:
            # 2D case:
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            if squeeze_happened:
                prob = np.expand_dims(prob, int(np.argwhere(prob_shape == 1))) # add back empty axis for compatibility
        elif prob.ndim == 3:
            # 3D case: 
            prob = transforms.resize_image(prob, Ly=to_y_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
            prob = transforms.resize_image(prob, Ly=to_z_size, Lx=to_x_size, no_channels=True)
            prob = prob.transpose(1, 0, 2)
        else:
            raise ValueError(f'gradients have incorrect dimension after squeezing. Should be 2 or 3, prob shape: {prob.shape}')
        
        return prob

    def _resize_gradients(self, grads: np.ndarray, to_y_size: int, to_x_size: int, to_z_size: int = None) -> np.ndarray:
        """
        Resize gradient arrays to specified dimensions for either 2D or 3D gradients.

        Parameters:
            grads (np.ndarray): The gradients to resize, either in 2D or 3D. Returns the same ndim as provided.
            to_y_size (int): The target size along the Y-axis.
            to_x_size (int): The target size along the X-axis.
            to_z_size (int, optional): The target size along the Z-axis. Required
                for 3D gradients.

        Returns:
            numpy.ndarray: The resized gradient array with the same number of dimensions
            as the input.

        Raises:
            ValueError: If the input gradient array does not have 3 or 4 dimensions.
        """
        grads_shape = grads.shape
        grads = grads.squeeze()
        squeeze_happened = grads.shape != grads_shape
        grads_shape = np.array(grads_shape)

        if grads.ndim == 3:
            # 2D case, with XY flows in 2 channels:
            grads = np.moveaxis(grads, 0, -1) # Put gradients last
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = np.moveaxis(grads, -1, 0) # Put gradients first

            if squeeze_happened:
                grads = np.expand_dims(grads, int(np.argwhere(grads_shape == 1))) # add back empty axis for compatibility
        elif grads.ndim == 4:
            # dP has gradients that can be treated as channels:
            grads = grads.transpose(1, 2, 3, 0) # move gradients last:
            grads = transforms.resize_image(grads, Ly=to_y_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(1, 0, 2, 3) # switch axes to resize again
            grads = transforms.resize_image(grads, Ly=to_z_size, Lx=to_x_size, no_channels=False)
            grads = grads.transpose(3, 1, 0, 2) # undo transposition
        else:
            raise ValueError(f'gradients have incorrect dimension after squeezing. Should be 3 or 4, grads shape: {grads.shape}')
        
        return grads
