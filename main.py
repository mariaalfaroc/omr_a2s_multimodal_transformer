import argparse
import os

from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image

from utils.kern import KrnConverter, ENCODING_OPTIONS
from data.grandstaff import load_gs_datasets, batch_preparation

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_distorted_images', action='store_true', help='Use distorted images')
    parser.add_argument('--fold_id', type=int, default=1, choices=[1, 2, 3, 4, 5], help='Fold id')
    parser.add_argument('--kern_encoding', type=str, default='bekern', choices=ENCODING_OPTIONS, help='Kern encoding')
    #parser.add_argument('--keep_ligatures', action='store_true', help='Keep ligatures')
    args = parser.parse_args()

    convKRN = KrnConverter()
    #convKRN = KrnConverter(keep_ligatures=False) # ERROR

    path = 'data/grandstaff/beethoven/piano-sonatas/sonata05-3/min3_down_m-90-95.bekrn'

    res = convKRN.encode(path)
    print(res)

    ## Checking all files:
    # base_path = 'data/grandstaff/'
    # with open('dst.txt', 'w') as fout:
    #     file_names = Path('data/grandstaff/pieces.txt').read_text().splitlines()
    #     for single_file in file_names:
    #         target_file = os.path.join(base_path, single_file)
    #         try:
    #             res = convKRN.encode(target_file)
    #             fout.write("{} - {}\n".format(target_file, "Done!"))
    #         except:
    #             fout.write("{} - {}\n".format(target_file, "Fail!"))

    CHECK_DIR = "check"
    if not os.path.isdir(CHECK_DIR):
        os.mkdir(CHECK_DIR)
   
    train_dataset, val_dataset, test_dataset = load_gs_datasets(path='data/grandstaff/mozart',
                                                                kern_encoding=args.kern_encoding,
                                                                use_distorted_images=args.use_distorted_images)
    
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, collate_fn=batch_preparation)

    print(f'Train dataset size: {len(train_dataset)}')
    for xa, xi, dec_in, dec_out in train_loader:
        print('Types:')
        print('\txa:', xa[0].dtype)
        print('\txi:', xi[0].dtype)
        print('\tdec_in:', dec_in[0].dtype)
        print('\tdec_out:', dec_out[0].dtype)
        print('Shapes:')
        print('\txa:', xa[0].shape)
        print('\txi:', xi[0].shape)
        print('\tdec_in:', dec_in[0].shape)
        print('\tdec_out:', dec_out[0].shape)

        # Save batch spectrogram/images
        save_image(make_grid(list(xa), nrow=4), f'{CHECK_DIR}/xa_train_batch.jpg')
        save_image(make_grid(list(xi), nrow=4), f'{CHECK_DIR}/xi_train_batch.jpg')

        # See first sample
        w2i, i2w = train_dataset.get_vocabulary()
        print(f'Shape with padding: {dec_in[0].shape}')
        print('Decoder input:', [i2w[i.item()] for i in dec_in[0]])
        print('Decoder output:', [i2w[i.item()] for i in dec_out[0]])
        save_image(xi[0], f'{CHECK_DIR}/xi0_train_batch.jpg')

        break