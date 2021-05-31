from options.test_options import TestOptions
from data import DataLoader
from models import create_model
from util.writer import Writer


def run_test(epoch=-1):
    print('Running Test')
    opt=dict(dataroot='datasets/human_seg' ,name='human_seg' ,arch='meshunet',
            dataset_mode='segmentation' ,ncf=[32 ,64 ,128 ,256],ninput_edges=2280,
            pool_res = [1800,1350,600] ,resblocks=3, batch_size =12,export_folder='meshes' )
    opt = TestOptions().parse()
    opt.serial_batches = True  # no shuffle
    dataset = DataLoader(opt)
    model = create_model(opt)
    writer = Writer(opt)
    # test
    writer.reset_counter()
    for i, data in enumerate(dataset):
        model.set_input(data)
        ncorrect, nexamples = model.test()
        writer.update_counter(ncorrect, nexamples)
    writer.print_acc(epoch, writer.acc)
    return writer.acc


if __name__ == '__main__':
    run_test()
