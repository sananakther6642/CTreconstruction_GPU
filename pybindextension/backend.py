import os
from torch.utils.cpp_extension import load

current_dir = os.path.dirname(os.path.abspath(__file__))


cpp_file_path = os.path.join(current_dir,  'src','add_test.cpp')


_backend = load(name='add_test',
                sources=[cpp_file_path],)

__all__ = ['_backend']
