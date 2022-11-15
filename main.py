import argparse
import os
import torch
from src.model.resnet import *
from src.train_model import prepare_trained_model
from src.directions import create_random_directions
from src.calc_loss import calulate_loss_landscape

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', default='resnet56', help='Type of networks')
    parser.add_argument('--logging_path', default='results', help='Base path of logging')
    parser.add_argument('--ex_name', default='resnet56_base', help='Type of Experiments')
    parser.add_argument('--model_path', default='', help='Path of pretrained weights')
    parser.add_argument('--best_model', type=bool, default=False, help='Load best or last epoch weights')
    # Determine the resolution
    parser.add_argument('--xmin', type=int, default=-1, help='Minimum value of x-coordinate')
    parser.add_argument('--xmax', type=int, default=1, help='Maximum value of x-coordinate')
    parser.add_argument('--xnum', type=int, default=51, help='Number of x-coordinate')
    parser.add_argument('--ymin', type=int, default=-1, help='Minimum value of y-coordinate')
    parser.add_argument('--ymax', type=int, default=1, help='Maximum value of y-coordinate')
    parser.add_argument('--ynum', type=int, default=51, help='Number of y-coordinate')
    # Save images
    parser.add_argument('--save_imges', type=bool, default=True, help='')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    # For 3D rendering
    parser.add_argument('--make_vtp', type=bool, default=True, help='')
    args = parser.parse_args()
    
    save_path = os.path.join(args.logging_path, args.ex_name) 
    if not os.path.isdir(save_path):
        os.makedirs(save_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.device = device
    
    if args.model_type == 'resnet56':
        model = ResNet56()
    else : 
        model = ResNet56_noshort()
        
    trained_model = prepare_trained_model(args, model, save_path)
    rand_directions = create_random_directions(trained_model)
    surface_path = calulate_loss_landscape(args, trained_model, rand_directions, save_path)
    if args.save_imges :
        from src.visualize import visualize
        visualize(args, save_path, surface_path)
    if args.make_vtp :
        from src.h52vtp import h5_to_vtp
        h5_to_vtp(surface_path)

