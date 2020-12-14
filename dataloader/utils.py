import pickle
from .dataloader import DataHandler, generate_batch
from torchtext.utils import download_from_url, extract_archive
from torch.utils.data import DataLoader

def prepare_data(train_path, val_path, test_path, dh_path, load_from_dump=True, bs=16):
    if load_from_dump == False:
        url_base = 'https://raw.githubusercontent.com/multi30k/dataset/master/data/task1/raw/'
        train_urls = ('train.de.gz', 'train.en.gz')
        val_urls = ('val.de.gz', 'val.en.gz')
        test_urls = ('test_2016_flickr.de.gz', 'test_2016_flickr.en.gz')

        train_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in train_urls]
        val_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in val_urls]
        test_filepaths = [extract_archive(download_from_url(url_base + url))[0] for url in test_urls]

        m_dh = DataHandler(train_filepaths)

        train_data = m_dh.data_process(train_filepaths)
        val_data = m_dh.data_process(val_filepaths)
        test_data = m_dh.data_process(test_filepaths)
        
        dump_data(train_data, train_path)
        dump_data(val_data, val_path)
        dump_data(test_data, test_path)
        dump_data(m_dh, dh_path)
    else:
        train_data = load_dump(train_path)
        val_data = load_dump(val_path)
        test_data = load_dump(test_path)
        m_dh = load_dump(dh_path)

    train_loader = DataLoader(train_data, batch_size=bs,
                            shuffle=True, collate_fn=generate_batch)
    valid_loader = DataLoader(val_data, batch_size=bs,
                            shuffle=False, collate_fn=generate_batch)
    test_loader = DataLoader(test_data, batch_size=bs,
                        shuffle=False, collate_fn=generate_batch)
    return train_loader, valid_loader, test_loader, m_dh

def dump_data(data, path):
    with open(path, 'wb') as f:
        pickle.dump(data, f)

def load_dump(path):
    with open(path, 'rb') as f:
        data = pickle.load(f)
    return data