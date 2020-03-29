
__all__ = ['get_model']

def get_model(params, dataset):
    if params['name'] == 'dense':
        from models.dense import Model
        return Model(params, dataset)
    assert False and 'Unknown model'

