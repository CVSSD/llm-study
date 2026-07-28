def add_to_class(Class):
    """egister function as method in created class"""
    def wrapper(obj):
        setattr(Class, obj.__name__, obj)
        print("Method", obj.__name__ ,"added to class ", Class.__name__)
    return wrapper

def remove_from_class(Class):
    def wrapper(obj):
        if hasattr(Class, obj.__name__):
            delattr(Class, obj.__name__)
            print("Method removed.")
        else:
            print("Method doesn't exist.")
    return wrapper

class HyperParameters:
    """The base class of hyperparameters."""
    def save_hyperparameters(self, ignore=[]):
        raise NotImplemented
