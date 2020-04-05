
__all__ = ['get_model']

def get_model(params, dataset):
    if params['name'] == 'dense':
        from models.dense import Model
        return Model(params, dataset)
    elif params['name'] == 'cnn1d':
        from models.cnn1d import Model
        return Model(params, dataset)
    assert False and 'Unknown model'

