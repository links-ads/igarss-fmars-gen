from pathlib import Path
import pandas as pd
import glob
import os


def get_region_name(event_name, metadata_root = './metadata'):
    """
    Get the region associate with the input event.
    It is based in the event_id2State2Region.csv file.
    Input:
        event_name: Example: 'southafrica-flooding22'
    Output:
        region name: Example: 'AfricaSouth-Full'
    """
    
    metadata_root = Path(metadata_root)
    df = pd.read_csv(metadata_root / 'evet_id2State2Region.csv')
    return df[df['event_id'] == event_name]['region'].values[0]


def get_all_events(data_root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data'):
    """
    Get all the events name in the data_root folder.
    Input:
        data_root: Example: '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data'
    Output:
        all_events: List of events.
    """
    
    data_root = Path(data_root)
    all_events = []
    for event_name in glob.glob('**/*.tif', recursive = True, root_dir=data_root):
        if event_name.split('/')[0] not in all_events:
            all_events.append(event_name.split('/')[0])
    return sorted(list(all_events))


def get_mosaics_names(event_name, data_root = '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data', when = None):
    """
    Get all the mosaic names for an event.
    Input:
        event_name: Example: 'Gambia-flooding-8-11-2022'
        data_root: Example: '/nfs/projects/overwatch/maxar-segmentation/maxar-open-data'
        when: 'pre' or 'post'. Default matches both
    Output:
        all_mosaics: List of mosaic names. Example: ['104001007A565700', '104001007A565800']
    """
    
    data_root = Path(data_root)
    all_mosaics = []
    if when in ['pre', 'post']:
        for mosaic_name in glob.glob('*', root_dir=data_root/event_name/when):
            all_mosaics.append(mosaic_name)
    elif when is None or when == 'None':
        for mosaic_name in glob.glob('**/*', root_dir=data_root/event_name):
            #all_mosaics.append(mosaic_name.split('/')[1])
            all_mosaics.append(os.path.split(mosaic_name)[1])
    else:
        raise ValueError('Variable when must be: "pre", "post" or "None"')
    return all_mosaics