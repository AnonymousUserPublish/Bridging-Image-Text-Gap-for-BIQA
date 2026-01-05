import torch
from typing import Union, Tuple, List, Dict, Any

def get_nested_shape_details(data: Any, name: str = "Root", indent: int = 0):
    """
    Recursively prints the structure and shape of a nested object 
    containing tuples, lists, dictionaries, and PyTorch tensors.
    """
    indent_str = "    " * indent
    
    if isinstance(data, (tuple, list)):
        print(f"{indent_str}├── {name} ({'Tuple' if isinstance(data, tuple) else 'List'}): Length {len(data)}")
        for i, item in enumerate(data):
            get_nested_shape_details(item, name=f"Element {i}", indent=indent + 1)
            
    elif isinstance(data, dict):
        print(f"{indent_str}├── {name} (Dictionary): Keys {list(data.keys())}")
        for key, value in data.items():
            get_nested_shape_details(value, name=f"Key '{key}'", indent=indent + 1)

    elif isinstance(data, torch.Tensor):
        shape_str = str(list(data.shape))
        dtype_str = str(data.dtype).replace("torch.", "")
        print(f"{indent_str}└── {name} (Tensor): Shape {shape_str}, Dtype {dtype_str}")
    
    elif data is None:
        print(f"{indent_str}└── {name} (None)")
        
    else:
        # For integers, floats, strings, etc.
        print(f"{indent_str}└── {name} (Other Type): Type {type(data).__name__}, Value Sample: {str(data)[:20]}")