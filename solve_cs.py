import argparse
from solvers.cs import solveCS


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='solve compressive sensing')
    parser.add_argument('-prior',type=str,help='choose with prior to use glow, dcgan, wavelet, dct', default='glow')
    parser.add_argument('-denoiser', type=str, help='choose your denoiser: nlm or bm3d', default='nlm')
    parser.add_argument('-sigma_f', type=float, help='choose your denoiser input noise level', default=3.0)
    parser.add_argument('-update_iter', type=int, help='choose frequency of red updates', default=1)
    parser.add_argument('-experiment', type=str, help='the name of the experiment',default='celeba_cs_glow')
    parser.add_argument('-dataset', type=str, help='the dataset/images to use',default='celeba')
    parser.add_argument('-model', type=str, help='which model to use',default='celeba')
    parser.add_argument('-m',  type=int, nargs='+',help='no. of measurements',default=[12288,10000,7500,5000,2500,1000,750,500,400,300,200,100,50,30,20])
    parser.add_argument('-gamma',  type=float, nargs='+',help='regularizor',default=[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
    parser.add_argument('-alpha', type=float, help='red multiplier', default=0.1)
    parser.add_argument('-beta', type=float, help='consistency multiplier', default=0.1)
    parser.add_argument('-multiplier', type=int, help='use multiplier decay schedule', default=0)
    parser.add_argument('-optim', type=str, help='optimizer', default="lbfgs")
    parser.add_argument('-lr', type=float, help='learning rate', default=0.1)
    parser.add_argument('-steps',type=int,help='no. of steps to run', default=30)
    parser.add_argument('-batchsize',type=int, help='no. of images to solve in parallel as batches',default=6) # batch size must be 2 for glowred prior
    parser.add_argument('-size',type=int, help='resize all images to this size', default=64)
    parser.add_argument('-device',type=str,help='device to use', default='cuda')
    parser.add_argument('-noise',type=str, help='noise to add. Either random_bora or float representing std of gaussian noise', default="random_bora")
    parser.add_argument('-init_strategy',type=str,help="init strategy to use",default='random')
    parser.add_argument('-init_std', type=float,help='std of init_strategy is random', default=0)
    parser.add_argument('-init_norms', type=float, nargs='+',help='initialization norm',default=None)
    parser.add_argument('-save_metrics_text',type=bool, help='whether to save results to a text file',default=True)
    parser.add_argument('-save_results',type=bool,help='whether to save results after experiments conclude',default=True)
    args = parser.parse_args()
    solveCS(args)
    
