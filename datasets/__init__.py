
__all__ = ['get_dataset']

def get_dataset(params):
    if params['name'] == 'multimodal_points':
        from datasets.multimodal_gaussian_2d import Dataset
        return Dataset(params)
    assert False and 'Unknown dataset'
