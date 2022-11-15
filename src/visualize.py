from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
import h5py
import numpy as np
import seaborn as sns
import os

def visualize(args, save_path, surface_path):
    result_file_path = os.path.join(save_path, '2D_images/')
    if not os.path.isdir(result_file_path):
        os.makedirs(result_file_path)
    surf_name = args.surf_name

    with h5py.File(surface_path,'r') as f:

        Z_LIMIT = 10

        x = np.array(f['xcoordinates'][:])
        y = np.array(f['ycoordinates'][:])

        X, Y = np.meshgrid(x, y)
        
        if surf_name in f.keys():
            Z = np.array(f[surf_name][:])
        elif surf_name == 'train_acc' or surf_name == 'test_acc' :
            Z = 100 - np.array(f[surf_name][:])
        else:
            print ('%s is not found in %s' % (surf_name, surface_path))
        
        Z = np.array(f[surf_name][:])
        #Z[Z > Z_LIMIT] = Z_LIMIT
        #Z = np.log(Z)  # logscale

        # Save 2D contours image
        fig = plt.figure()
        CS = plt.contour(X, Y, Z, cmap = 'summer', levels=np.arange(args.vmin, args.vmax, args.vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + surf_name + '_2dcontour' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        fig = plt.figure()
        CS = plt.contourf(X, Y, Z, levels=np.arange(args.vmin, args.vmax, args.vlevel))
        plt.clabel(CS, inline=1, fontsize=8)
        fig.savefig(result_file_path + surf_name + '_2dcontourf' + '.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

        # Save 2D heatmaps image
        plt.figure()
        sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=args.vmin, vmax=args.vmax,
                               xticklabels=False, yticklabels=False)
        sns_plot.invert_yaxis()
        sns_plot.get_figure().savefig(result_file_path + surf_name + '_2dheat.pdf',
                                      dpi=300, bbox_inches='tight', format='pdf')

        # Save 3D surface image
        plt.figure()
        ax = Axes3D(fig)
        ax.plot_surface(X, Y, Z, linewidth=0, antialiased=True)
        fig.savefig(result_file_path + surf_name + '_3dsurface.pdf', dpi=300,
                    bbox_inches='tight', format='pdf')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--logging_path', default='results', help='Base path of logging')
    parser.add_argument('--ex_name', default='resnet56_base', help='Type of Experiments')
    parser.add_argument('--surf_name', default='train_loss', help='The type of surface to plot')
    parser.add_argument('--vmin', default=0.1, type=float, help='Miminum value to map')
    parser.add_argument('--vmax', default=10, type=float, help='Maximum value to map')
    parser.add_argument('--vlevel', default=0.5, type=float, help='plot contours every vlevel')
    args = parser.parse_args()
    save_path = os.path.join(args.logging_path, args.ex_name) 
    surface_path = f"{save_path}/3d_surface_file.h5"
    visualize(args, save_path, surface_path)